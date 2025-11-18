"""
================================================================================
Train Faster R-CNN for Furniture Part + Damage Detection 
================================================================================
Key Improvements:
 - Uses pretrained backbone (COCO) for stronger generalization
 - Correct classification head size and mapping
 - Resizes + color jitter only (keeps bbox geometry correct)
 - Train/val split, validation loss, and mAP evaluation
 - Automatic training curve + mAP curve saved
 - AMP mixed precision for fast GPU training
 - Clean collate_fn and training loop
 - Reproducible training
================================================================================
"""

import os
import json
import random
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
#  DEFINE CLASSES
# -----------------------------
PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]
CLASSES = ["__background__"] + PARTS + DAMAGES
NAME2IDX = {name: idx for idx, name in enumerate(CLASSES)}


# -----------------------------
#  DATASET
# -----------------------------
class FurnitureDataset(Dataset):
    def __init__(self, ann_path, img_dir, resize=512):
        with open(ann_path, "r") as f:
            self.ann = json.load(f)

        self.img_dir = img_dir
        self.resize = resize

        self.tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img = Image.open(os.path.join(self.img_dir, a["filename"])).convert("RGB")

        orig_w, orig_h = img.size
        sx = self.resize / float(orig_w)
        sy = self.resize / float(orig_h)

        boxes = []
        labels = []

        for pname, box in a["parts"].items():
            if pname in PARTS:
                boxes.append(box)
                labels.append(NAME2IDX[pname])

        for d in a.get("damages", []):
            part = d["part"]
            dtype = d["type"]
            if part in a["parts"] and dtype in DAMAGES:
                boxes.append(a["parts"][part])
                labels.append(NAME2IDX[dtype])

        scaled = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]

        target = {
            "boxes": torch.tensor(scaled, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        return self.tf(img), target


# -----------------------------
#  MODEL
# -----------------------------
def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
#  mAP EVALUATION
# -----------------------------
def compute_map(model, val_loader, device, iou_threshold=0.5):
    model.eval()
    aps = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)

            for pred, tgt in zip(preds, targets):
                if len(pred["boxes"]) == 0 or len(tgt["boxes"]) == 0:
                    continue

                ious = box_iou(pred["boxes"].cpu(), tgt["boxes"].cpu())
                max_iou_vals, max_iou_idx = ious.max(dim=1)

                tp = sum(
                    (max_iou_vals >= iou_threshold)
                    & (pred["labels"].cpu() == tgt["labels"][max_iou_idx].cpu())
                )
                fp = len(pred["boxes"]) - tp
                fn = len(tgt["boxes"]) - tp

                ap = tp / (tp + fp + fn + 1e-6)
                aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


# -----------------------------
#  TRAINING LOOP
# -----------------------------
def train_model(model, train_loader, val_loader, device, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler(device="cuda") if device == "cuda" else None

    train_losses, val_losses, map_scores = [], [], []

    for epoch in range(num_epochs):
        # -------- TRAIN --------
        model.train()
        train_loss_sum = 0.0

        for imgs, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    loss_dict = model(imgs, targets)
                    loss = sum(loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item()

        epoch_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(epoch_train_loss)

        # --------------------- VALIDATION ---------------------
        model.eval()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{num_epochs}"):
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                val_loss_sum += loss.item()
        
        epoch_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(epoch_val_loss)
        

        # -------- mAP --------
        epoch_map = compute_map(model, val_loader, device)
        map_scores.append(epoch_map)

        print(f"[Epoch {epoch+1}] Train={epoch_train_loss:.4f}  Val={epoch_val_loss:.4f}  mAP={epoch_map:.4f}")

    # ----- SAVE PLOTS -----
    os.makedirs("./models/damage_detection", exist_ok=True)

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.savefig("./models/damage_detection/training_curve.png")
    plt.close()

    # mAP curve
    plt.figure(figsize=(7, 4))
    plt.plot(map_scores, label="mAP@0.5")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP Curve")
    plt.savefig("./models/damage_detection/map_curve.png")
    plt.close()

    return model


# -----------------------------
#  MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default="./data/synthetic_damage/annotations.json")
    parser.add_argument("--imgs", default="./data/synthetic_damage/images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--resize", type=int, default=512)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = FurnitureDataset(args.ann, args.imgs, args.resize)

    # Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training on {device}. Samples={len(dataset)}")

    model = build_model(len(CLASSES)).to(device)

    model = train_model(model, train_loader, val_loader, device, args.epochs)

    save_path = "./models/damage_detection/frcnn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Saved model to {save_path}")


if __name__ == "__main__":
    main()
