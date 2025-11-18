import os
import json
import random
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteHead, SSDLiteClassificationHead, SSDLiteRegressionHead

# -----------------------------
# CLASSES
# -----------------------------
PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES
NAME2IDX = {name: idx for idx, name in enumerate(CLASSES)}

# -----------------------------
# Dataset
# -----------------------------
class ChairDataset(Dataset):
    def __init__(self, ann_path, img_dir, resize=320):
        with open(ann_path, "r") as f:
            self.ann = json.load(f)

        self.img_dir = img_dir
        self.resize = resize

        self.tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img = Image.open(os.path.join(self.img_dir, a["filename"])).convert("RGB")
        orig_w, orig_h = img.size

        sx = self.resize / orig_w
        sy = self.resize / orig_h

        boxes = []
        labels = []

        # parts
        for pname, b in a["parts"].items():
            if pname in PARTS:
                x1, y1, x2, y2 = b
                boxes.append([x1*sx, y1*sy, x2*sx, y2*sy])
                labels.append(NAME2IDX[pname])

        # damages
        for d in a["damages"]:
            p = d["part"]
            t = d["type"]
            if p in a["parts"] and t in DAMAGES:
                x1, y1, x2, y2 = a["parts"][p]
                boxes.append([x1*sx, y1*sy, x2*sx, y2*sy])
                labels.append(NAME2IDX[t])

        if len(boxes) == 0:
            boxes = [[0,0,5,5]]
            labels = [0]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return self.tf(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# BUILD MODEL
# -----------------------------
def build_model(num_classes):
    """
    Build an SSDLite model for custom number of classes.
    The model needs to be properly configured for the number of classes.
    """
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
    
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    
    # Get the in_channels for each feature level
    in_channels = [module[1].in_channels for module in model.head.classification_head.module_list]
    
    # Get the num_anchors per location for each feature level
    # For SSDLite, we have 6 anchors per location
    num_anchors = [6] * len(in_channels)
    
    # Use GroupNorm instead of BatchNorm to avoid issues with small batch sizes
    norm_layer = lambda num_channels: torch.nn.GroupNorm(min(32, num_channels // 4 if num_channels >= 4 else 1), num_channels)
    
    # Replace the classification and regression heads with ones configured for our num_classes
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer
    )
    model.head.regression_head = SSDLiteRegressionHead(
        in_channels, num_anchors, norm_layer=norm_layer
    )
    return model



# -----------------------------
# TRAIN
# -----------------------------
def train(model, train_loader, val_loader, device, epochs):
    # Freeze backbone parameters and set to eval mode to avoid batch norm issues
    model.backbone.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.head.train()
        t_loss = 0

        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        avg_train = t_loss / len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.head.train()  # Keep head in train mode to get loss dict
        v_loss = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = [img.to(device) for img in imgs]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
                loss_dict = model(imgs, tgts)
                v_loss += sum(loss_dict.values()).item()

        avg_val = v_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"[Epoch {epoch+1}] Train={avg_train:.4f}  Val={avg_val:.4f}")

    # Save curves
    os.makedirs("./models/damage_detection", exist_ok=True)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("./models/damage_detection/mobilenet_training_curve.png")

    return model


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default="./data/synthetic_damage/annotations.json")
    parser.add_argument("--imgs", default="./data/synthetic_damage/images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = ChairDataset(args.ann, args.imgs)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, args.batch, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(len(CLASSES)).to(device)
    model = train(model, train_loader, val_loader, device, args.epochs)

    torch.save(model.state_dict(), "./models/damage_detection/mobilenet_ssd.pth")
    print("[OK] Saved model to mobilenet_ssd.pth")


if __name__ == "__main__":
    main()
