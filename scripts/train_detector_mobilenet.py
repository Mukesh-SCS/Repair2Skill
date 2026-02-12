"""
Train parts-only detector (SSDLite MobileNetV3) + damage classifier (ResNet18).
Uses synth_v2 annotations: parts (boxes) + part_damage (per-part damage label).

Usage:
  # Generate data first: python scripts/generate_synthetic_data.py --output_dir ./data/synth_v2 --samples 12000
  # Train both (default):
  python scripts/train_detector_mobilenet.py --task all
  # Train parts detector only:
  python scripts/train_detector_mobilenet.py --task parts --epochs_parts 80
  # Train damage classifier only:
  python scripts/train_detector_mobilenet.py --task damage --epochs_damage 30

Outputs:
  models/damage_detection/parts_detector_ssd.pth
  models/damage_detection/damage_classifier_resnet18.pth
  models/damage_detection/parts_training_graph.png  (if --task parts or all)
  models/damage_detection/damage_training_graph.png  (if --task damage or all)
"""

import os
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models import resnet18, MobileNet_V3_Large_Weights
from torchvision.ops import box_iou

PARTS = [
    "seat", "back",
    "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg",
    "armrest_left", "armrest_right",
]
DAMAGE_TYPES = ["none", "missing", "cracked", "broken", "loose", "scratched"]

PART_CLASSES = ["__background__"] + PARTS
PART_NAME2IDX = {n: i for i, n in enumerate(PART_CLASSES)}
DMG_NAME2IDX = {n: i for i, n in enumerate(DAMAGE_TYPES)}


def _resize_pad_320(img: Image.Image):
    orig_w, orig_h = img.size
    target = 320
    tmp = img.copy()
    tmp.thumbnail((target, target), Image.Resampling.LANCZOS)
    new_w, new_h = tmp.size
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    out = Image.new("RGB", (target, target), (128, 128, 128))
    out.paste(tmp, (pad_x, pad_y))
    sx = new_w / orig_w
    sy = new_h / orig_h
    return out, (sx, sy, pad_x, pad_y)


class PartsDataset(Dataset):
    def __init__(self, ann_path, img_dir, augment=True):
        self.ann = json.load(open(ann_path, "r", encoding="utf-8"))
        self.img_dir = img_dir
        self.augment = augment

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img = Image.open(os.path.join(self.img_dir, a["filename"])).convert("RGB")

        if self.augment and random.random() < 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        if self.augment and random.random() < 0.4:
            img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))

        img320, (sx, sy, px, py) = _resize_pad_320(img)
        img_t = TF.to_tensor(img320)
        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        boxes = []
        labels = []
        for p, box in a["parts"].items():
            if p not in PARTS:
                continue
            x1, y1, x2, y2 = box
            x1 = x1 * sx + px
            x2 = x2 * sx + px
            y1 = y1 * sy + py
            y2 = y2 * sy + py
            x1 = max(0, min(319, x1))
            x2 = max(x1 + 2, min(320, x2))
            y1 = max(0, min(319, y1))
            y2 = max(y1 + 2, min(320, y2))
            boxes.append([x1, y1, x2, y2])
            labels.append(PART_NAME2IDX[p])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_t, target


def collate(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)


def _compute_ap_voc(recalls, precisions):
    """VOC-style AP: 11-point interpolation (average max precision at 0,0.1,...,1)."""
    if not recalls or not precisions or sum(recalls) == 0:
        return 0.0
    ap = 0.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_at_t = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t:
                p_at_t = max(p_at_t, p)
        ap += p_at_t / 11.0
    return ap


def compute_map50(model, val_dl, device, score_thresh=0.25, iou_thresh=0.5, num_classes=9):
    """Compute mAP@0.5 on validation set (background = 0)."""
    model.eval()
    # Per-class: list of (confidence, is_tp)
    class_scores_tp = [[] for _ in range(num_classes)]
    class_num_gt = [0] * num_classes

    with torch.no_grad():
        for imgs, tgts in val_dl:
            imgs = [i.to(device) for i in imgs]
            preds = model(imgs)

            for pred, tgt in zip(preds, tgts):
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)
                keep = pred["scores"] >= score_thresh
                p_boxes = pred["boxes"][keep].to(device)
                p_scores = pred["scores"][keep]
                p_labels = pred["labels"][keep]
                for c in range(1, num_classes):
                    n_gt = (gt_labels == c).sum().item()
                    class_num_gt[c] += n_gt
                    pred_c = (p_labels == c).nonzero(as_tuple=True)[0]
                    if len(pred_c) == 0:
                        continue
                    pb = p_boxes[pred_c]
                    ps = p_scores[pred_c]
                    gb = gt_boxes[gt_labels == c]
                    if gb.numel() == 0:
                        for s in ps.tolist():
                            class_scores_tp[c].append((s, False))
                        continue
                    ious = box_iou(pb, gb)
                    used_gt = set()
                    for ord_idx in torch.argsort(ps, descending=True):
                        row = ious[ord_idx]
                        best_gt = row.argmax().item()
                        iou_val = row[best_gt].item()
                        tp = iou_val >= iou_thresh and best_gt not in used_gt
                        if tp:
                            used_gt.add(best_gt)
                        class_scores_tp[c].append((ps[ord_idx].item(), tp))
    aps = []
    for c in range(1, num_classes):
        if class_num_gt[c] == 0:
            continue
        lst = class_scores_tp[c]
        if not lst:
            aps.append(0.0)
            continue
        lst.sort(key=lambda x: -x[0])
        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        for _, is_tp in lst:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / max(1, tp_cum + fp_cum))
            recalls.append(tp_cum / max(1, class_num_gt[c]))
        ap = _compute_ap_voc(recalls, precisions)
        aps.append(ap)
    return sum(aps) / max(1, len(aps))


class DamageCropDataset(Dataset):
    """Uses ground-truth part boxes to generate crops and classify damage type per part."""

    def __init__(self, ann_path, img_dir, augment=True):
        self.ann = json.load(open(ann_path, "r", encoding="utf-8"))
        self.img_dir = img_dir
        self.augment = augment
        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.samples = []
        for a in self.ann:
            pd = a.get("part_damage", {})
            for p in PARTS:
                if p not in a["parts"]:
                    continue
                dmg = pd.get(p, "none")
                self.samples.append((a["filename"], p, a["parts"][p], dmg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, p, box, dmg = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))

        if self.augment and random.random() < 0.4:
            crop = TF.adjust_brightness(crop, random.uniform(0.85, 1.15))
        if self.augment and random.random() < 0.4:
            crop = TF.adjust_contrast(crop, random.uniform(0.85, 1.15))
        if self.augment and random.random() < 0.2:
            crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        x = self.tf(crop)
        y = torch.tensor(DMG_NAME2IDX[dmg], dtype=torch.long)
        return x, y


def train_parts_detector(train_ann, val_ann, img_dir, out_path, epochs=60, batch=16, lr=1e-4, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_ds = PartsDataset(train_ann, img_dir, augment=True)
    val_ds = PartsDataset(val_ann, img_dir, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

    # Pretrained backbone (ImageNet); new head for 9 classes (background + 8 parts)
    try:
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            num_classes=len(PART_CLASSES),
        )
    except TypeError:
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=len(PART_CLASSES))
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_map = 0.0
    history = {"train_loss": [], "val_loss": [], "mAP50": []}
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for imgs, tgts in train_dl:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss.item())
        tl /= max(1, len(train_dl))
        scheduler.step()

        model.train()
        vl = 0.0
        with torch.no_grad():
            for imgs, tgts in val_dl:
                imgs = [i.to(device) for i in imgs]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
                vl += float(loss.item())
        vl /= max(1, len(val_dl))

        mAP50 = compute_map50(model, val_dl, device, num_classes=len(PART_CLASSES))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["mAP50"].append(mAP50)
        print(f"[PART DET] ep={ep} train_loss={tl:.3f} val_loss={vl:.3f} mAP@0.5={mAP50:.3f}")

        if mAP50 > best_map:
            best_map = mAP50
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": PART_CLASSES}, out_path)
            print(f"[SAVE] {out_path} (mAP@0.5={mAP50:.3f})")

    # Training curves
    out_dir = os.path.dirname(out_path)
    if out_dir and history["train_loss"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ep_x = list(range(1, len(history["train_loss"]) + 1))
        axes[0].plot(ep_x, history["train_loss"], label="Train loss", color="C0")
        axes[0].set_title("Parts detector – Train loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(ep_x, history["val_loss"], label="Val loss", color="C1")
        axes[1].set_title("Parts detector – Val loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[2].plot(ep_x, history["mAP50"], label="mAP@0.5", color="C2")
        axes[2].set_title("Parts detector – mAP@0.5")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        plt.tight_layout()
        graph_path = os.path.join(out_dir, "parts_training_graph.png")
        plt.savefig(graph_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[GRAPH] Saved {graph_path}")


def train_damage_classifier(train_ann, val_ann, img_dir, out_path, epochs=25, batch=64, lr=3e-4, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_ds = DamageCropDataset(train_ann, img_dir, augment=True)
    val_ds = DamageCropDataset(val_ann, img_dir, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    # Class distribution and optional weighting
    class_counts = [0] * len(DAMAGE_TYPES)
    for _, _, _, dmg in train_ds.samples:
        class_counts[DMG_NAME2IDX[dmg]] += 1
    print("[DMG CLS] Train class counts:", dict(zip(DAMAGE_TYPES, class_counts)))
    total = sum(class_counts)
    class_weights = None
    nonzero = [c for c in class_counts if c > 0]
    if nonzero and total > 0 and max(nonzero) / max(1, min(nonzero)) > 2:
        class_weights = torch.tensor(
            [total / (len(DAMAGE_TYPES) * max(1, c)) for c in class_counts],
            dtype=torch.float32,
            device=device,
        )
        class_weights = class_weights / class_weights.mean()
        print("[DMG CLS] Using class weights (imbalance > 2x):", [f"{w:.2f}" for w in class_weights.tolist()])

    model = resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(DAMAGE_TYPES))
    model.to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss.item())
        tl /= max(1, len(train_dl))

        model.eval()
        correct = 0
        total_n = 0
        cm = [[0] * len(DAMAGE_TYPES) for _ in range(len(DAMAGE_TYPES))]
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total_n += int(y.numel())
                for gt, pr in zip(y.tolist(), pred.tolist()):
                    cm[gt][pr] += 1
        acc = correct / max(1, total_n)
        history["train_loss"].append(tl)
        history["val_acc"].append(acc)
        print(f"[DMG CLS] ep={ep} train_loss={tl:.3f} val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "damage_types": DAMAGE_TYPES}, out_path)
            print(f"[SAVE] {out_path}")

    # Training curves
    out_dir = os.path.dirname(out_path)
    if out_dir and history["train_loss"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ep_x = list(range(1, len(history["train_loss"]) + 1))
        axes[0].plot(ep_x, history["train_loss"], label="Train loss", color="C0")
        axes[0].set_title("Damage classifier – Train loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(ep_x, history["val_acc"], label="Val accuracy", color="C1")
        axes[1].set_title("Damage classifier – Val accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        plt.tight_layout()
        graph_path = os.path.join(out_dir, "damage_training_graph.png")
        plt.savefig(graph_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[GRAPH] Saved {graph_path}")

    # Confusion matrix (on validation)
    print("[DMG CLS] Validation confusion matrix (rows=GT, cols=pred):")
    print("       " + " ".join(f"{t:>8}" for t in DAMAGE_TYPES))
    for i, name in enumerate(DAMAGE_TYPES):
        row = " ".join(f"{cm[i][j]:>8}" for j in range(len(DAMAGE_TYPES)))
        print(f"{name:>6} {row}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", default="./data/synth_v2/annotations_train.json")
    ap.add_argument("--val_ann", default="./data/synth_v2/annotations_val.json")
    ap.add_argument("--img_dir", default="./data/synth_v2/images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", choices=["parts", "damage", "all"], default="all")
    ap.add_argument("--epochs_parts", type=int, default=60)
    ap.add_argument("--epochs_damage", type=int, default=25)
    ap.add_argument("--batch_parts", type=int, default=16)
    ap.add_argument("--batch_damage", type=int, default=64)
    args = ap.parse_args()

    if args.task in ("parts", "all"):
        train_parts_detector(
            args.train_ann, args.val_ann, args.img_dir,
            out_path="./models/damage_detection/parts_detector_ssd.pth",
            epochs=args.epochs_parts, batch=args.batch_parts, device=args.device,
        )

    if args.task in ("damage", "all"):
        train_damage_classifier(
            args.train_ann, args.val_ann, args.img_dir,
            out_path="./models/damage_detection/damage_classifier_resnet18.pth",
            epochs=args.epochs_damage, batch=args.batch_damage, device=args.device,
        )
