"""
================================================================================
Detect Parts + Damages using Faster R-CNN 
================================================================================
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

PART_CLASSES = [
    "seat","back","front_left_leg","front_right_leg",
    "back_left_leg","back_right_leg","armrest_left","armrest_right"
]

DAMAGE_CLASSES = ["missing","cracked","broken","loose","scratched"]
ALL_CLASSES = PART_CLASSES + DAMAGE_CLASSES  # total 13
NUM_CLASSES = len(ALL_CLASSES) + 1           # + background


# -----------------------------
# BUILD MODEL
# -----------------------------
def build_model(weights_path):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # pretrained backbone
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


# -----------------------------
# DETECTION PIPELINE
# -----------------------------
@torch.no_grad()
def detect_damage_and_parts(image_path, weights, threshold=0.2, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(weights).to(device)
    tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    out = model(x)[0]  # predictions
    boxes, scores, labels = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()

    parts = []
    damages = []

    for b, s, l in zip(boxes, scores, labels):
        if s < threshold:
            continue

        idx = int(l.item()) - 1   # remove background offset
        if idx < 0 or idx >= len(ALL_CLASSES):
            continue

        name = ALL_CLASSES[idx]
        box = b.tolist()

        if name in PART_CLASSES:
            parts.append({"part": name, "box": box, "confidence": float(s)})
        else:
            damages.append({"type": name, "box": box, "confidence": float(s)})

    # --------------------------
    # PAIR PARTS ↔ DAMAGES
    # --------------------------
    pairs = []
    if parts and damages:
        pb = torch.tensor([p["box"] for p in parts])
        db = torch.tensor([d["box"] for d in damages])
        ious = box_iou(pb, db)

        for i, p in enumerate(parts):
            if torch.max(ious[i]) > 0.30:
                j = int(torch.argmax(ious[i]))
            else:
                px1, py1, px2, py2 = pb[i]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2

                dists = []
                for j_, d in enumerate(damages):
                    dx1, dy1, dx2, dy2 = db[j_]
                    dcx, dcy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
                    dists.append(((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5)
                j = int(torch.tensor(dists).argmin())

            pairs.append({
                "part": parts[i]["part"],
                "damage_type": damages[j]["type"],
                "part_confidence": parts[i]["confidence"],
                "damage_confidence": damages[j]["confidence"]
            })

    result = {
        "detected_parts": parts,
        "detected_damages": damages,
        "detected_pairs": pairs
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage1_parts.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
