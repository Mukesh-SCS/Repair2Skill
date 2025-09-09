"""
================================================================================
DESCRIPTION:
    Trains a Faster R-CNN detector on the synthetic annotations to localize
    damaged parts. Saves a PyTorch state_dict for inference.

USAGE:
    python scripts/train_detector_frcnn.py \
      --ann ./data/synthetic_damage/annotations.json \
      --imgs ./data/synthetic_damage/images \
      --epochs 10 --batch 2

OUTPUTS:
    ./models/damage_detection/frcnn_model.pth

ARGUMENTS:
    --ann     Path to annotations.json
    --imgs    Path to images folder
    --epochs  Number of epochs (default 10)
    --batch   Batch size (default 2)
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os
import json
import argparse

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FurnitureDetectionDataset(Dataset):
    def __init__(self, annotations_file: str, images_dir: str, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        # background + 8 parts + 5 damages = 14 classes
        self.class_names = [
            "__background__", "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right",
            "missing", "cracked", "broken", "loose", "scratched"
        ]
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        rec = self.annotations[idx]
        img_path = os.path.join(self.images_dir, rec["filename"])
        image = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        parts = rec.get("parts", {})

        # build one box per damaged part using its part bounding box
        for damage in rec.get("damages", []):
            part = damage["part"]
            label = damage["type"]
            if part in parts:
                x1, y1, x2, y2 = parts[part]
                boxes.append([x1, y1, x2, y2])
                labels.append(self.name_to_idx[label])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image = self.transforms(image)
        return image, target


def get_transform():
    return T.Compose([T.ToTensor()])


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to annotations.json")
    ap.add_argument("--imgs", required=True, help="Path to images folder")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    dataset = FurnitureDetectionDataset(args.ann, args.imgs, transforms=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 14  # background + 13
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(args.epochs):
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs} - loss: {float(loss):.4f}")

    os.makedirs("./models/damage_detection", exist_ok=True)
    torch.save(model.state_dict(), "./models/damage_detection/frcnn_model.pth")
    print("Saved ./models/damage_detection/frcnn_model.pth")


if __name__ == "__main__":
    main()
