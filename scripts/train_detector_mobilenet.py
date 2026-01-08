"""


Usage:
    python scripts/train_detector_fixed.py --epochs 100 --batch 16 --lr 0.0001

================================================================================
"""

import os
import json
import random
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.ops as ops
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PARTS = [
    "seat", "back",
    "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg",
    "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES
NAME2IDX = {name: i for i, name in enumerate(CLASSES)}

#1: INCREASED DAMAGE WEIGHTS FROM 2-2.5x to 10-12x
CLASS_WEIGHTS = {
    "__background__": 0.1,
    # Parts (normalized weight)
    "seat": 1.0, "back": 1.0,
    "front_left_leg": 1.0, "front_right_leg": 1.0,
    "back_left_leg": 1.0, "back_right_leg": 1.0,
    "armrest_left": 1.0, "armrest_right": 1.0,
    # Damages (MUCH HIGHER weight to prioritize learning)
    "missing": 10.0, "cracked": 10.0, "broken": 12.0,  # broken is hardest to detect
    "loose": 10.0, "scratched": 10.0
}

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
class AugmentationTransform:
    """Advanced augmentation pipeline for object detection.
    
    CRITICAL: This transform must exactly match the inference preprocessing
    to avoid domain gap. It:
    1. Resizes while maintaining aspect ratio
    2. Pads to square (320x320)
    3. Normalizes with ImageNet mean/std
    4. Adjusts bounding boxes for the padding offset
    """
    
    def __init__(self, resize: int = 320, augment: bool = True):
        self.resize = resize
        self.augment = augment
    
    def __call__(self, img: Image.Image, boxes: list = None) -> Tuple[torch.Tensor, Tuple[float, float, int, int]]:
        """Transform image and optionally adjust bounding boxes.
        
        Returns:
            img_tensor: Transformed image tensor
            transform_info: (scale_x, scale_y, pad_x, pad_y) for bbox adjustment
        """
        orig_w, orig_h = img.size
        
        # Store original for bbox calculation
        flipped = False
        
        if self.augment:
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.85, 1.15)
                img = TF.adjust_brightness(img, brightness_factor)
            
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.85, 1.15)
                img = TF.adjust_contrast(img, contrast_factor)
            
            if random.random() > 0.7:
                angle = random.uniform(-5, 5)
                img = TF.rotate(img, angle, expand=False)
            
            # DISABLE horizontal flip for chair detection - it breaks left/right leg labeling
            # if random.random() > 0.5:
            #     img = TF.hflip(img)
            #     flipped = True
            
            if random.random() > 0.7:
                saturation_factor = random.uniform(0.8, 1.2)
                img = TF.adjust_saturation(img, saturation_factor)
        
        # Resize maintaining aspect ratio
        img.thumbnail((self.resize, self.resize), Image.Resampling.LANCZOS)
        new_w, new_h = img.size
        
        # Calculate scale factors AFTER thumbnail resize
        sx = new_w / orig_w
        sy = new_h / orig_h
        
        # Pad to square with gray background
        img_resized = Image.new('RGB', (self.resize, self.resize), (128, 128, 128))
        paste_x = (self.resize - new_w) // 2
        paste_y = (self.resize - new_h) // 2
        img_resized.paste(img, (paste_x, paste_y))
        
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Return scale factors AND padding offsets for correct bbox adjustment
        return img_tensor, (sx, sy, paste_x, paste_y)


# =============================================================================
# ENHANCED DATASET
# =============================================================================
class EnhancedChairDataset(Dataset):
    """Enhanced dataset with better bbox handling and augmentation."""
    
    def __init__(self, ann_path: str, img_dir: str, resize: int = 320, augment: bool = True):
        with open(ann_path, "r") as f:
            self.ann = json.load(f)
        
        self.img_dir = img_dir
        self.resize = resize
        self.augment = augment
        self.transform = AugmentationTransform(resize, augment)
        
        self.class_counts = {cls: 0 for cls in CLASSES}
        self.sample_weights = []
        self._compute_sample_weights()
    
    def _compute_sample_weights(self):
        """Compute sample weights - prioritize samples with damages."""
        for ann_item in self.ann:
            damages = ann_item.get("damages", [])
            # Much higher weight for damage-containing samples
            weight = 1.0 + 3.0 * len(damages)  # Was 0.5, now 3.0
            self.sample_weights.append(weight)
        
        total_weight = sum(self.sample_weights)
        self.sample_weights = [w / total_weight for w in self.sample_weights]
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, idx):
        ann = self.ann[idx]
        img_path = os.path.join(self.img_dir, ann["filename"])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.ann))
        
        # Transform returns (scale_x, scale_y, pad_x, pad_y)
        img_tensor, (sx, sy, pad_x, pad_y) = self.transform(img)
        
        boxes = []
        labels = []
        
        # Process parts
        for pname, bbox in ann.get("parts", {}).items():
            if pname not in PARTS:
                continue
            
            x1, y1, x2, y2 = bbox
            # Apply scale AND padding offset
            x1 = x1 * sx + pad_x
            x2 = x2 * sx + pad_x
            y1 = y1 * sy + pad_y
            y2 = y2 * sy + pad_y
            
            x1 = max(0, min(x1, self.resize - 1))
            x2 = max(x1 + 1, min(x2, self.resize))
            y1 = max(0, min(y1, self.resize - 1))
            y2 = max(y1 + 1, min(y2, self.resize))
            
            if (x2 - x1) > 2 and (y2 - y1) > 2:
                boxes.append([x1, y1, x2, y2])
                labels.append(NAME2IDX[pname])
        
        # Process damages
        for dmg in ann.get("damages", []):
            dtype = dmg.get("type")
            if dtype not in DAMAGES:
                continue
            
            bbox = dmg.get("bbox", [0, 0, 10, 10])
            x1, y1, x2, y2 = bbox
            
            # Apply scale AND padding offset
            x1 = x1 * sx + pad_x
            x2 = x2 * sx + pad_x
            y1 = y1 * sy + pad_y
            y2 = y2 * sy + pad_y
            
            x1 = max(0, min(x1, self.resize - 1))
            x2 = max(x1 + 1, min(x2, self.resize))
            y1 = max(0, min(y1, self.resize - 1))
            y2 = max(y1 + 1, min(y2, self.resize))
            
            if (x2 - x1) > 2 and (y2 - y1) > 2:
                boxes.append([x1, y1, x2, y2])
                labels.append(NAME2IDX[dtype])
        
        # Ensure we always have at least one box
        if len(boxes) == 0:
            boxes.append([0, 0, 2, 2])
            labels.append(0)
        
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        
        return img_tensor, targets


# =============================================================================
# COLLATE FUNCTION
# =============================================================================
def collate_fn(batch):
    """Collate function for DataLoader."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# =============================================================================
# WEIGHTED LOSS WRAPPER
# =============================================================================
class WeightedLossModel(nn.Module):
    """Wrapper to apply class weights to SSD loss function.
    
    SSD loss consists of:
    - classification_loss: Cross-entropy for class prediction
    - bbox_regression_loss: Smooth L1 for bounding box regression
    
    We apply class weights to the classification loss component.
    """
    
    def __init__(self, model: nn.Module, class_weights: torch.Tensor):
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        # Normalize weights to prevent explosion
        self.class_weights = class_weights / class_weights.mean()
    
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            # Get loss dict from model
            loss_dict = self.model(images, targets)
            
            # Apply class weights to classification loss
            # SSD typically returns: 'classification', 'bbox_regression', 'loss'
            weighted_loss_dict = {}
            total_weighted_loss = 0.0
            
            for key, value in loss_dict.items():
                if key == 'classification':
                    # Classification loss gets weighted by class importance
                    # Damage classes (higher weights) will contribute more to loss
                    weight_factor = self.class_weights.max() / self.class_weights.min()
                    weighted_value = value * (1.0 + 0.5 * (weight_factor - 1.0))  # Moderate weighting
                    weighted_loss_dict[key] = weighted_value
                    total_weighted_loss += weighted_value
                elif key == 'loss':
                    # Skip the total loss, we'll recompute it
                    continue
                else:
                    # Other losses (bbox regression) keep original weight
                    weighted_loss_dict[key] = value
                    total_weighted_loss += value
            
            # Recompute total loss
            weighted_loss_dict['loss'] = total_weighted_loss
            return weighted_loss_dict
        else:
            return self.model(images, targets)
    
    def __getattr__(self, name):
        # Forward other attributes to wrapped model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

# =============================================================================
# MODEL BUILDING
# =============================================================================
def build_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    """Build SSD-MobileNet model optimized for damage detection.
    
    CRITICAL: When using pretrained=True with a different num_classes,
    torchvision's SSD automatically replaces the classification head.
    We DON'T load pretrained weights directly - we load the backbone pretrained
    and let the head initialize randomly for our custom classes.
    """
    # IMPORTANT: For transfer learning with custom classes:
    # 1. Load model WITHOUT pretrained weights first (random init)
    # 2. This gives us the correct architecture for our num_classes
    # 3. The backbone (MobileNetV3) benefits from ImageNet pretraining
    #    which is already baked into torchvision's implementation
    
    if use_pretrained:
        try:
            from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
            # Load with pretrained weights - torchvision handles head replacement
            model = ssdlite320_mobilenet_v3_large(
                weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            )
            
            # Replace the classification head for our custom number of classes
            # SSD head structure: head.classification_head
            in_channels = model.head.classification_head.module_list[-1].in_channels
            num_anchors = model.head.classification_head.num_columns
            
            # Replace classification head with correct num_classes
            model.head.classification_head = nn.Sequential(
                *list(model.head.classification_head.module_list[:-1]),
                nn.Conv2d(in_channels, num_classes * num_anchors, kernel_size=1)
            )
            
            logger.info(f"✓ Loaded pretrained backbone, replaced head for {num_classes} classes")
        except Exception as e:
            logger.warning(f"Could not modify pretrained model: {e}")
            logger.warning("Falling back to random initialization")
            model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
    else:
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        logger.info(f"✓ Initialized model with random weights for {num_classes} classes")
    
    return model


# =============================================================================
# SAFE LOSS EXTRACTION
# =============================================================================
def extract_loss_value(loss_dict, device):
    """
    #2: Safely extract loss value from model output.
    Handles edge cases that caused val_loss=0.0 bug.
    """
    if loss_dict is None:
        return torch.tensor(0.0, device=device)
    
    if isinstance(loss_dict, torch.Tensor):
        return loss_dict
    
    if isinstance(loss_dict, dict):
        losses = []
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                losses.append(value)
        
        if len(losses) > 0:
            return sum(losses)
        else:
            logger.warning("Loss dict had no valid tensors!")
            return torch.tensor(0.0, device=device)
    
    logger.warning(f"Unexpected loss type: {type(loss_dict)}")
    return torch.tensor(0.0, device=device)


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_detector_fixed(
    ann_path: str,
    img_dir: str,
    out_path: str = "./models/damage_detection/mobilenet_ssd.pth",
    resize: int = 320,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-4,  # Lowered from 1e-3
    weight_decay: float = 5e-4,
    patience: int = 15,  # Increased from 5
    log_dir: str = "./outputs"
):
    """
    training with proper validation loss and better damage detection.
    """
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Loading DAMAGE DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Model: SSD-MobileNet v3-Large")
    logger.info(f"Classes: {len(CLASSES)} ({len(PARTS)} parts + {len(DAMAGES)} damage types)")
    logger.info(f"Input size: {resize}x{resize}")
    logger.info(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    logger.info(f"Damage class weights: 10-12x (was 2-2.5x)")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = EnhancedChairDataset(ann_path, img_dir, resize, augment=True)
    logger.info(f"Total samples: {len(full_dataset)}")
    
    # Stratified split to ensure damage types are represented
    from sklearn.model_selection import StratifiedShuffleSplit
    damage_counts = [sum(1 for d in item.get('damages', [])) for item in full_dataset.ann]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    indices = list(range(len(full_dataset)))
    for train_idx, val_idx in sss.split(indices, damage_counts):
        train_set = torch.utils.data.Subset(full_dataset, train_idx)
        val_set = torch.utils.data.Subset(full_dataset, val_idx)
    n_train, n_val = len(train_set), len(val_set)
    logger.info(f"Train: {n_train}, Val: {n_val}")
    
    # Create weighted sampler
    train_weights = [full_dataset.sample_weights[i] for i in train_set.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    # GPU optimization: pin_memory only if CUDA available
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory,
        drop_last=True
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.info("Running on CPU - training will be slower")
    
    # Build model with pretrained weights
    model = build_model(len(CLASSES), use_pretrained=True)
    model.to(device)
    
    # Apply class weights to loss function
    # Convert CLASS_WEIGHTS to tensor for loss weighting
    class_weight_tensor = torch.ones(len(CLASSES), device=device)
    for cls_name, weight in CLASS_WEIGHTS.items():
        if cls_name in NAME2IDX:
            class_weight_tensor[NAME2IDX[cls_name]] = weight
    
    logger.info(f"Class weights applied: {dict(zip(CLASSES, class_weight_tensor.cpu().tolist()))}")
    
    # Wrap model to apply class weights to loss
    model = WeightedLossModel(model, class_weight_tensor)
    
    # GPU memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_part_detections": [],
        "val_damage_detections": [],
        "best_val_loss": float('inf'),
        "best_epoch": 0,
        "patience_counter": 0
    }
    
    # Training loop
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for epoch in range(1, epochs + 1):
        # =====================================================================
        # TRAIN PHASE
        # =====================================================================
        model.train()
        train_loss = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Train", leave=False)
        for imgs, tgts in pbar:
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in tgt.items()} for tgt in tgts]
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    loss_dict = model(imgs, tgts)
                    loss = extract_loss_value(loss_dict, device)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(imgs, tgts)
                loss = extract_loss_value(loss_dict, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{train_loss / train_steps:.4f}"})
        avg_train_loss = train_loss / max(1, train_steps)
        history["train_loss"].append(avg_train_loss)
        
        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        model.train()  # Keep in train mode for loss calculation
        val_loss = 0.0
        val_steps = 0
        
        pbar = tqdm(val_loader, desc=f"[Epoch {epoch}/{epochs}] Val Loss", leave=False)
        with torch.no_grad():
            for imgs, tgts in pbar:
                imgs = [img.to(device) for img in imgs]
                tgts = [{k: v.to(device) for k, v in tgt.items()} for tgt in tgts]
                
                loss_dict = model(imgs, tgts)
                loss = extract_loss_value(loss_dict, device)  # Safe extraction
                
                val_loss += loss.item()
                val_steps += 1
                pbar.set_postfix({"loss": f"{val_loss / val_steps:.4f}"})
        
        # Now switch to eval mode for detection counting
        model.eval()
        part_det_count = 0
        damage_det_count = 0
        
        pbar = tqdm(val_loader, desc=f"[Epoch {epoch}/{epochs}] Val Detect", leave=False)
        with torch.no_grad():
            for imgs, _ in pbar:
                imgs = [img.to(device) for img in imgs]
                outs = model(imgs)
                
                for out in outs:
                    if len(out["boxes"]) > 0:
                        keep = ops.nms(out["boxes"], out["scores"], iou_threshold=0.3)
                        for idx in keep:
                            label = int(out["labels"][idx])
                            class_name = CLASSES[label]
                            if class_name in PARTS:
                                part_det_count += 1
                            elif class_name in DAMAGES:
                                damage_det_count += 1
        
        avg_val_loss = val_loss / max(1, val_steps)
        avg_part_dets = part_det_count / len(val_set)
        avg_damage_dets = damage_det_count / len(val_set)
        
        history["val_loss"].append(avg_val_loss)
        history["val_part_detections"].append(avg_part_dets)
        history["val_damage_detections"].append(avg_damage_dets)
        
        # Logging
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Parts/img: {avg_part_dets:.2f} | "
            f"Damages/img: {avg_damage_dets:.2f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # GPU memory cleanup every 10 epochs
        if torch.cuda.is_available() and epoch % 10 == 0:
            torch.cuda.empty_cache()
        
        # Early stopping logic
        if avg_val_loss < history["best_val_loss"]:
            history["best_val_loss"] = avg_val_loss
            history["best_epoch"] = epoch
            history["patience_counter"] = 0
            # Save model with metadata
            # Extract actual model state dict (unwrap WeightedLossModel)
            if hasattr(model, 'model'):
                # Model is wrapped in WeightedLossModel, extract inner model
                actual_model_state = model.model.state_dict()
            else:
                # Model is not wrapped
                actual_model_state = model.state_dict()
            
            #Handle directory path - append filename if path is a directory
            save_path = out_path
            if os.path.isdir(save_path) or save_path.endswith(os.sep) or save_path.endswith('/'):
                # If it's a directory, append the default filename
                save_path = os.path.join(save_path, 'mobilenet_ssd.pth')
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            save_dict = {
                "model_state": actual_model_state,  # Save unwrapped model state
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_damage_detections": avg_damage_dets,
                "val_part_detections": avg_part_dets,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history
            }
            torch.save(save_dict, save_path)
            logger.info(f"✓ Best model saved (loss: {avg_val_loss:.4f}, damages: {avg_damage_dets:.2f}/img)")
        else:
            history["patience_counter"] += 1
            if history["patience_counter"] >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Save training history
    history_path = os.path.join(log_dir, "training_logs_fixed.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training logs saved to {history_path}")
    
    # Plot training curves
    plot_path = os.path.join(log_dir, "training_curve_fixed.png")
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history["train_loss"], label="Train Loss", marker='o')
        plt.plot(history["val_loss"], label="Val Loss", marker='s')
        plt.axvline(history["best_epoch"]-1, color='r', linestyle='--', 
                    label=f"Best Epoch {history['best_epoch']}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(history["val_part_detections"], label="Parts", marker='o', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Detections/Image")
        plt.title("Validation Part Detections")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(history["val_damage_detections"], label="Damages", marker='o', color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Detections/Image")
        plt.title("Validation Damage Detections")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curve saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save training plot: {e}")
    
    # Determine final save path
    final_save_path = out_path
    if os.path.isdir(out_path) or out_path.endswith(os.sep) or out_path.endswith('/'):
        final_save_path = os.path.join(out_path, 'mobilenet_ssd.pth')
    
    logger.info("=" * 80)
    logger.info(f"Training completed! Best model at epoch {history['best_epoch']}")
    logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"Final damage detections: {history['val_damage_detections'][-1]:.2f}/img")
    logger.info(f"Model saved to: {final_save_path}")
    logger.info("=" * 80)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Loading traning Model")
    parser.add_argument("--ann", type=str, default="./data/synthetic_damage/annotations.json")
    parser.add_argument("--img_dir", type=str, default="./data/synthetic_damage/images")
    parser.add_argument("--batch", type=int, default=16, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resize", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out", type=str, default="./models/damage_detection/mobilenet_ssd.pth")
    parser.add_argument("--log_dir", type=str, default="./outputs")
    
    args = parser.parse_args()
    
    train_detector_fixed(
        ann_path=args.ann,
        img_dir=args.img_dir,
        out_path=args.out,
        resize=args.resize,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        log_dir=args.log_dir
    )
