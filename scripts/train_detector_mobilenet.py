import os
import json
import random
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.ops as ops
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# CLASSES (update if you change dataset)
# ---------------------------------------------------------
PARTS = [
    "seat", "back",
    "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg",
    "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["background"] + PARTS + DAMAGES
NAME2IDX = {name: i for i, name in enumerate(CLASSES)}


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class ChairDataset(Dataset):
    def __init__(self, ann_path: str, img_dir: str, resize: int = 320):
        with open(ann_path, "r") as f:
            self.ann = json.load(f)

        self.img_dir = img_dir
        self.resize = resize

        # Correct, distortion-free resizing
        self.tf = transforms.Compose([
            transforms.Resize(resize),                   # keeps aspect ratio
            transforms.CenterCrop((resize, resize)),     # square crop
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img_path = os.path.join(self.img_dir, a["filename"])
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size

        # scaling from Resize(resize)
        sx = self.resize / orig_w
        sy = self.resize / orig_h

        # resized height (after Resize)
        new_h = int(orig_h * sy)
        crop_pad = (self.resize - new_h) // 2  # CenterCrop padding

        boxes = []
        labels = []

        # PARTS
        for pname, box in a["parts"].items():
            if pname not in PARTS:
                continue
            x1, y1, x2, y2 = box

            x1 = x1 * sx
            x2 = x2 * sx
            y1 = y1 * sy + crop_pad
            y2 = y2 * sy + crop_pad

            boxes.append([x1, y1, x2, y2])
            labels.append(NAME2IDX[pname])

        # DAMAGES
        for d in a["damages"]:
            t = d["type"]
            if t not in DAMAGES:
                continue

            x1, y1, x2, y2 = d["bbox"]
            x1 = x1 * sx
            x2 = x2 * sx
            y1 = y1 * sy + crop_pad
            y2 = y2 * sy + crop_pad

            boxes.append([x1, y1, x2, y2])
            labels.append(NAME2IDX[t])

        if len(boxes) == 0:
            boxes = [[0, 0, 5, 5]]
            labels = [0]

        tgt = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return self.tf(img), tgt


# ---------------------------------------------------------
# COLLATE
# ---------------------------------------------------------
def collate(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)


# ---------------------------------------------------------
# SSD MODEL BUILDER
# ---------------------------------------------------------

def build_model(num_classes: int):
    """
    Build an SSD model with custom number of classes.
    Uses the standard SSD with MobileNet V3 backbone but adapts it for custom classes.
    """
    from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
    
    try:
        # Load the pretrained SSD-MobileNet model
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return model
    except TypeError:
        # Fallback if num_classes parameter is not supported in this version
        # Load with default classes and then adapt
        model = ssdlite320_mobilenet_v3_large(weights=None)
        
        # Get the expected number of classes
        num_anchors_list = model.anchor_generator.num_anchors_per_location()
        
        # Replace classification head
        old_num_classes = model.head.classification_head.num_classes
        ratio = num_classes / old_num_classes
        
        new_cls_head = nn.Sequential()
        old_cls_head = model.head.classification_head
        
        # Rebuild classification head with new output channels
        cls_modules = nn.ModuleList()
        for module in old_cls_head.module_list:
            # Each module is a Sequential with depthwise and pointwise convolutions
            new_module = nn.Sequential()
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    if layer.out_channels != layer.in_channels:  # This is the pointwise conv
                        # Replace with new pointwise conv with updated output channels
                        new_out_channels = int(layer.out_channels * ratio)
                        new_conv = nn.Conv2d(
                            layer.in_channels,
                            new_out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding
                        )
                        new_module.add_module(str(len(new_module)), new_conv)
                    else:
                        new_module.add_module(str(len(new_module)), layer)
                else:
                    new_module.add_module(str(len(new_module)), layer)
            cls_modules.append(new_module)
        
        model.head.classification_head.module_list = cls_modules
        model.head.classification_head.num_classes = num_classes
        
        return model




# ---------------------------------------------------------
# TRAINING LOOP (with DAMAGE WEIGHTING)
# ---------------------------------------------------------
def train_detector(
    ann_path: str,
    img_dir: str,
    out_path: str = "./models/damage_detection/mobilenet_ssd.pth",
    resize: int = 320,
    batch: int = 16,
    epochs: int = 10,
    lr: float = 1e-4,
    damage_weight: float = 5.0
):

    print("[INFO] Loading dataset...")
    full = ChairDataset(ann_path, img_dir, resize)
    N = len(full)
    n_val = int(N * 0.2)
    n_train = N - n_val

    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False,
                            collate_fn=collate, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device:", device)

    model = build_model(len(CLASSES))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, tgts in tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]"):
            imgs = [im.to(device) for im in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            loss_dict = model(imgs, tgts)

            # Default SSD outputs:
            # loss_dict["classification"], loss_dict["bbox_regression"]

            cls_loss = loss_dict.get("classification", torch.tensor(0.0, device=device))
            box_loss = loss_dict.get("bbox_regression", torch.tensor(0.0, device=device))

            # DAMAGE WEIGHTING
            weighted_cls = cls_loss * damage_weight

            loss = weighted_cls + box_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss = {running_loss / len(train_loader):.4f}")

        # -----------------------------------------------------
        # Validation sanity check
        # -----------------------------------------------------
        model.eval()
        det_count = 0
        cls_freq = {}
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = [im.to(device) for im in imgs]

                outs = model(imgs)

                for out in outs:
                    keep = ops.nms(out["boxes"], out["scores"], 0.3)
                    for i in keep:
                        lbl = int(out["labels"][i])
                        cls_freq[CLASSES[lbl]] = cls_freq.get(CLASSES[lbl], 0) + 1
                        det_count += 1

        avg_det = det_count / len(val_set)
        top = sorted(cls_freq.items(), key=lambda x: -x[1])[:6]

        print(f"[VAL] avg detections/img: {avg_det:.2f}")
        print("[VAL] top predicted classes:", top)

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print("[OK] Saved model to", out_path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ann", type=str, default="./data/synthetic_damage/annotations.json")
    p.add_argument("--img_dir", type=str, default="./data/synthetic_damage/images")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--resize", type=int, default=320)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", type=str, default="./models/damage_detection/mobilenet_ssd.pth")
    args = p.parse_args()

    train_detector(
        ann_path=args.ann,
        img_dir=args.img_dir,
        out_path=args.out,
        resize=args.resize,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr
    )
