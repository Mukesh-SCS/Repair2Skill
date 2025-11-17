"""
================================================================================
Train Faster R-CNN for Furniture Part + Damage Detection 
================================================================================
Key Improvements:
 - Uses pretrained backbone (COCO) for stronger generalization
 - Correct classification head size and mapping
 - Randomized geometric augmentations for synthetic chairs
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# -----------------------------
#  DEFINE CLASSES
# -----------------------------
PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = [
    "missing", "cracked", "broken", "loose", "scratched"
]

CLASSES = ["__background__"] + PARTS + DAMAGES   # total = 1 + 8 + 5 = 14
NAME2IDX = {name: idx for idx, name in enumerate(CLASSES)}


# -----------------------------
#  DATASET
# -----------------------------
class FurnitureDataset(Dataset):
    def __init__(self, ann_path, img_dir, resize=512):
        with open(ann_path, "r") as f:
            self.ann = json.load(f)

        self.img_dir = img_dir

        self.tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomRotation(3),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img = Image.open(os.path.join(self.img_dir, a["filename"])).convert("RGB")

        boxes = []
        labels = []

        # Load parts
        for pname, box in a["parts"].items():
            if pname in PARTS:
                boxes.append(box)
                labels.append(NAME2IDX[pname])

        # Load damages
        for d in a.get("damages", []):
            part = d["part"]
            dtype = d["type"]
            if part in a["parts"] and dtype in DAMAGES:
                boxes.append(a["parts"][part])
                labels.append(NAME2IDX[dtype])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return self.tf(img), target


# -----------------------------
#  MODEL BUILDER
# -----------------------------
def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
#  TRAINING LOOP
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

    ds = FurnitureDataset(args.ann, args.imgs, args.resize)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(len(CLASSES)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)

    print(f"[INFO] Training on {device}. Classes={len(CLASSES)} Samples={len(ds)}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for imgs, tgts in tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[EPOCH {epoch+1}] Loss = {total_loss / len(dl):.4f}")

    os.makedirs("./models/damage_detection", exist_ok=True)
    save_path = "./models/damage_detection/frcnn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Saved model to {save_path}")


if __name__ == "__main__":
    main()
