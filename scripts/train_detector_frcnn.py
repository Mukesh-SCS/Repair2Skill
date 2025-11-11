"""
================================================================================
Train Faster R-CNN for Furniture Damage Detection
================================================================================
Usage:
  python scripts/train_detector_frcnn.py \
      --ann ./data/synthetic_damage/annotations.json \
      --imgs ./data/synthetic_damage/images \
      --epochs 20 --batch 4
Outputs:
  ./models/damage_detection/frcnn_model.pth
================================================================================
"""

import os, json, argparse, random, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

PARTS   = ["seat","back","front_left_leg","front_right_leg","back_left_leg","back_right_leg","armrest_left","armrest_right"]
DAMAGES = ["missing","cracked","broken","loose","scratched"]
CLASSES = ["__background__"] + PARTS + DAMAGES
NAME2IDX = {n:i for i,n in enumerate(CLASSES)}

class FurnitureDetectionDataset(Dataset):
    def __init__(self, ann_path, img_dir, resize=512, max_items=None, part_dropout=0.4):
        with open(ann_path,"r") as f: self.ann = json.load(f)
        if max_items: self.ann = self.ann[:max_items]
        self.img_dir = img_dir
        self.tf = transforms.Compose([transforms.Resize((resize,resize)), transforms.ToTensor()])
        self.part_dropout = part_dropout

    def __len__(self): return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img = Image.open(os.path.join(self.img_dir, a["filename"])).convert("RGB")
        boxes, labels = [], []
        parts = a.get("parts", {})

        # parts (drop some to balance)
        for p, box in parts.items():
            if p in PARTS and random.random() > self.part_dropout:
                boxes.append(box); labels.append(NAME2IDX[p])

        # damages (always keep)
        for d in a.get("damages", []):
            p, t = d["part"], d["type"]
            if p in parts and t in DAMAGES:
                boxes.append(parts[p]); labels.append(NAME2IDX[t])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return self.tf(img), target

def build_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, len(CLASSES))
    return model

def collate_fn(b): return tuple(zip(*b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="./data/synthetic_damage/annotations.json")
    ap.add_argument("--imgs", default="./data/synthetic_damage/images")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--part-dropout", type=float, default=0.4)
    args = ap.parse_args()

    ds = FurnitureDetectionDataset(args.ann, args.imgs, args.resize, args.max_images, args.part_dropout)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

    print(f"[INFO] Training {len(ds)} samples, epochs={args.epochs}, batch={args.batch}, device={device}")
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for imgs, tgts in tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = [i.to(device) for i in imgs]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            opt.zero_grad(); 
            loss.backward(); 
            opt.step()
            total += loss.detach().item()
        print(f"Epoch {epoch+1}: loss={total/len(dl):.4f}")

    os.makedirs("./models/damage_detection", exist_ok=True)
    torch.save(model.state_dict(), "./models/damage_detection/frcnn_model.pth")
    print("[OK] saved ./models/damage_detection/frcnn_model.pth")

if __name__ == "__main__":
    main()
