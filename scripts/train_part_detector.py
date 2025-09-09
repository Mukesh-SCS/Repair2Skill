"""
================================================================================
DESCRIPTION:
    Trains a MobileNetV3-based multi-label classifier to predict:
    - damage types present in the image,
    - chair parts present in the image.

USAGE (standalone CLI):
    python scripts/train_part_detector.py \
      --ann ./data/synthetic_damage/annotations.json \
      --imgs ./data/synthetic_damage/images \
      --epochs 20 --batch 32

USAGE (import from main.py):
    from scripts.train_part_detector import train_model
    train_model()  # uses defaults below, no argparse involved

OUTPUTS:
    ./models/damage_detection/checkpoints/epoch_*.pth
    ./models/damage_detection/part_detector.pth
    ./outputs/training_curves.png
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os, json, time, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import matplotlib.pyplot as plt

DAMAGE_TYPES = ["missing", "cracked", "broken", "loose", "scratched"]
PART_TYPES = ["seat","back","front_left_leg","front_right_leg","back_left_leg","back_right_leg","armrest_left","armrest_right"]

class FurnitureDataset(Dataset):
    def __init__(self, annotations_file: str, images_dir: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        self.damage_to_idx = {d: i for i, d in enumerate(DAMAGE_TYPES)}
        self.part_to_idx = {p: i for i, p in enumerate(PART_TYPES)}
    def __len__(self): return len(self.annotations)
    def __getitem__(self, idx):
        rec = self.annotations[idx]
        img = Image.open(os.path.join(self.images_dir, rec["filename"])).convert("RGB")
        if self.transform: img = self.transform(img)
        d_lbl = torch.zeros(len(DAMAGE_TYPES)); p_lbl = torch.zeros(len(PART_TYPES))
        for d in rec.get("damages", []):
            d_lbl[self.damage_to_idx[d["type"]]] = 1.0
            p_lbl[self.part_to_idx[d["part"]]] = 1.0
        return img, d_lbl, p_lbl

class FurnitureRepairModel(nn.Module):
    def __init__(self, num_damage_classes=5, num_part_classes=8):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        feat_dim = 576
        self.damage_classifier = nn.Sequential(nn.Linear(feat_dim,256), nn.ReLU(), nn.Dropout(0.5),
                                               nn.Linear(256, num_damage_classes), nn.Sigmoid())
        self.part_classifier   = nn.Sequential(nn.Linear(feat_dim,256), nn.ReLU(), nn.Dropout(0.5),
                                               nn.Linear(256, num_part_classes), nn.Sigmoid())
    def forward(self, x):
        feats = self.backbone(x)
        return self.damage_classifier(feats), self.part_classifier(feats)

def train_model(ann="./data/synthetic_damage/annotations.json",
                imgs="./data/synthetic_damage/images",
                epochs: int = 20,
                batch: int = 32):
    """Library entrypoint. Safe to call from main.py; no argparse here."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = FurnitureDataset(ann, imgs, transform=tf)
    n_train = int(0.8 * len(ds)); n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4)

    model = FurnitureRepairModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    os.makedirs("./models/damage_detection/checkpoints", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)

    train_losses, val_losses = [], []
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tr_loss = 0.0
        for imgs_b, d_lbl, p_lbl in train_loader:
            imgs_b, d_lbl, p_lbl = imgs_b.to(device), d_lbl.to(device), p_lbl.to(device)
            optimizer.zero_grad()
            d_out, p_out = model(imgs_b)
            loss = criterion(d_out, d_lbl) + criterion(p_out, p_lbl)
            loss.backward(); optimizer.step()
            tr_loss += float(loss)
        tr_loss /= max(1,len(train_loader))

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for imgs_b, d_lbl, p_lbl in val_loader:
                imgs_b, d_lbl, p_lbl = imgs_b.to(device), d_lbl.to(device), p_lbl.to(device)
                d_out, p_out = model(imgs_b)
                loss = criterion(d_out, d_lbl) + criterion(p_out, p_lbl)
                va_loss += float(loss)
        va_loss /= max(1,len(val_loader))
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(va_loss)
        print(f"Epoch {ep+1}/{epochs} - Train {tr_loss:.4f}  Val {va_loss:.4f}")
        if (ep+1) % 5 == 0:
            torch.save(model.state_dict(), f"./models/damage_detection/checkpoints/epoch_{ep+1}.pth")

    torch.save(model.state_dict(), "./models/damage_detection/part_detector.pth")
    print(f"Saved ./models/damage_detection/part_detector.pth in {time.time()-t0:.1f}s")

    # plot losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig("./outputs/training_curves.png"); plt.close()

# Standalone CLI still supported
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann",  default="./data/synthetic_damage/annotations.json")
    parser.add_argument("--imgs", default="./data/synthetic_damage/images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch",  type=int, default=32)
    a = parser.parse_args()
    train_model(ann=a.ann, imgs=a.imgs, epochs=a.epochs, batch=a.batch)
