"""
================================================================================
Train Faster R-CNN for Furniture Damage Detection (Optimized)
================================================================================
Usage:
  python scripts/train_detector_frcnn.py \
      --ann ./data/synthetic_damage/annotations.json \
      --imgs ./data/synthetic_damage/images \
      --epochs 10 --batch 4 --resize 512 --workers 4

Outputs:
  ./models/damage_detection/frcnn_model.pth
================================================================================
"""

import os
import json
import argparse
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class FurnitureDetectionDataset(Dataset):
    def __init__(self, ann_path, img_dir, resize=512, max_items=None):
        with open(ann_path, "r") as f:
            self.annotations = json.load(f)
        if max_items:
            self.annotations = self.annotations[:max_items]
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        self.classes = [
            "__background__", "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right",
            "missing", "cracked", "broken", "loose", "scratched",
        ]
        self.name_to_idx = {n: i for i, n in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        a = self.annotations[idx]
        img = Image.open(os.path.join(self.img_dir, a["filename"])).convert("RGB")
        boxes, labels = [], []
        parts = a.get("parts", {})
        for d in a.get("damages", []):
            part, typ = d["part"], d["type"]
            if part in parts:
                x1, y1, x2, y2 = parts[part]
                boxes.append([x1, y1, x2, y2])
                labels.append(self.name_to_idx[typ])
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return self.transform(img), target


# -----------------------------------------------------------------------------
# Model Builder
# -----------------------------------------------------------------------------
def build_model(num_classes=14, arch="resnet50"):
    if arch == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="./data/synthetic_damage/annotations.json")
    ap.add_argument("--imgs", default="./data/synthetic_damage/images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--arch", choices=["resnet50", "mobilenet"], default="resnet50")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    args = ap.parse_args()

    cudnn.benchmark = True  # optimize conv algorithms

    dataset = FurnitureDetectionDataset(args.ann, args.imgs, resize=args.resize, max_items=args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    num_classes = len(dataset.classes)
    model = build_model(num_classes, arch=args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # try compile for speed (PyTorch ≥ 2.0)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"[INFO] Starting training for {args.epochs} epochs on {device} "
          f"(batch={args.batch}, workers={args.workers})")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for step, (imgs, targets) in enumerate(pbar, start=1):
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast(enabled=torch.cuda.is_available()):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values()) / args.accum_steps

            scaler.scale(loss).backward()

            if step % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach().item()
            pbar.set_postfix({"Batch Loss": f"{float(loss):.4f}"})

        mean_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Mean Loss: {mean_loss:.4f}")

    os.makedirs("./models/damage_detection", exist_ok=True)
    torch.save(model.state_dict(), "./models/damage_detection/frcnn_model.pth")
    print("[OK] Model saved to ./models/damage_detection/frcnn_model.pth")


if __name__ == "__main__":
    main()
