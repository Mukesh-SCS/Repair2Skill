"""
================================================================================
Detect Damage and Part Type using Faster R-CNN
================================================================================
Outputs:
  {
    "detected_damages": [...],
    "detected_parts": [...],
    "detected_pairs": [...]
  }

Usage:
  from scripts.detect_damage import detect_damage_and_parts
  detect_damage_and_parts("chair.jpg", "./models/damage_detection/frcnn_model.pth")
================================================================================
"""

import os
import torch
from torchvision import transforms
from PIL import Image

PART_CLASSES = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]
DAMAGE_CLASSES = ["missing", "cracked", "broken", "loose", "scratched"]


def detect_damage_and_parts(image_path: str, model_path: str, threshold: float = 0.6):
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    # Load model
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    num_classes = 1 + len(PART_CLASSES) + len(DAMAGE_CLASSES)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Prepare image
    tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    x = tf(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        predictions = model(x)[0]

    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]

    detected_parts, detected_damages, pairs = [], [], []
    for label, score in zip(labels, scores):
        if score < threshold:
            continue
        label_name = (
            PART_CLASSES[label - 1]
            if label - 1 < len(PART_CLASSES)
            else DAMAGE_CLASSES[label - 1 - len(PART_CLASSES)]
        )
        if label_name in PART_CLASSES:
            detected_parts.append({"part": label_name, "confidence": float(score)})
        else:
            detected_damages.append({"type": label_name, "confidence": float(score)})

    # Pair top damage with top part
    if detected_parts and detected_damages:
        best_damage = max(detected_damages, key=lambda x: x["confidence"])
        best_part = max(detected_parts, key=lambda x: x["confidence"])
        pairs.append({
            "part": best_part["part"],
            "damage_type": best_damage["type"],
            "damage_confidence": best_damage["confidence"],
            "part_confidence": best_part["confidence"],
        })

    return {
        "detected_damages": detected_damages,
        "detected_parts": detected_parts,
        "detected_pairs": pairs,
    }
