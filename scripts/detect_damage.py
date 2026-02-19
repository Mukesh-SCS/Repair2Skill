"""
Chair damage detection: parts detector (SSDLite) + damage classifier (ResNet18).
Detects parts → crops each part → classifies damage per part.
Returns parts, best_damage, and detected_pairs (for pipeline compatibility).
"""

import json
import os

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models import resnet18
from torchvision.ops import nms

PARTS = [
    "seat", "back",
    "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg",
    "armrest_left", "armrest_right",
]
PART_CLASSES = ["__background__"] + PARTS
DAMAGE_TYPES = ["none", "missing", "cracked", "broken", "loose", "scratched"]


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


def _load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_parts_detector(weights_path, device):
    ckpt = _load_ckpt(weights_path, device)
    model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=len(PART_CLASSES))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model


def load_damage_classifier(weights_path, device):
    ckpt = _load_ckpt(weights_path, device)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(DAMAGE_TYPES))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model


_damage_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load-once cache for server/repeated inference (key = (parts_path, damage_path))
_cached_models = {}
_cached_device = None


def get_models(parts_weights, damage_weights, device, force_reload=False):
    """Load parts detector and damage classifier once; reuse for subsequent calls."""
    global _cached_models, _cached_device
    key = (os.path.abspath(parts_weights), os.path.abspath(damage_weights))
    if force_reload or key not in _cached_models or _cached_device != device:
        _cached_models[key] = (
            load_parts_detector(parts_weights, device),
            load_damage_classifier(damage_weights, device),
        )
        _cached_device = device
    return _cached_models[key]


def detect_damage(
    image_path,
    parts_weights="./models/damage_detection/parts_detector_ssd.pth",
    damage_weights="./models/damage_detection/damage_classifier_resnet18.pth",
    part_thresh=0.35,
    max_parts=8,
    force_reload=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts_model, dmg_model = get_models(parts_weights, damage_weights, device, force_reload=force_reload)

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    img320, (sx, sy, px, py) = _resize_pad_320(img)
    x = TF.to_tensor(img320)
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    # Torchvision detection models expect a list of tensors (one per image), not a batched tensor
    with torch.no_grad():
        out = parts_model([x])[0]

    boxes = out["boxes"].detach().cpu()
    scores = out["scores"].detach().cpu()
    labels = out["labels"].detach().cpu()

    keep = scores >= part_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.numel() == 0:
        return {
            "image_path": image_path,
            "parts": [],
            "best_damage": None,
            "detected_pairs": [],
        }

    keep_idx = nms(boxes, scores, iou_threshold=0.4)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    parts = []
    n_keep = min(max_parts, len(boxes))
    for i in range(n_keep):
        b = boxes[i]
        s = scores[i]
        l = labels[i]
        cls = PART_CLASSES[int(l)]
        x1, y1, x2, y2 = b.numpy().tolist()
        x1 = (x1 - px) / sx
        x2 = (x2 - px) / sx
        y1 = (y1 - py) / sy
        y2 = (y2 - py) / sy
        x1 = float(max(0, min(orig_w, x1)))
        x2 = float(max(0, min(orig_w, x2)))
        y1 = float(max(0, min(orig_h, y1)))
        y2 = float(max(0, min(orig_h, y2)))

        crop = img.crop((x1, y1, x2, y2))
        cx = _damage_tf(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = dmg_model(cx)
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        dmg_idx = int(prob.argmax())
        dmg_type = DAMAGE_TYPES[dmg_idx]
        dmg_conf = float(prob[dmg_idx])

        part_confidence = float(s.item())
        smart_score = part_confidence * dmg_conf
        parts.append({
            "part": cls,
            "part_confidence": part_confidence,
            "bbox": [x1, y1, x2, y2],
            "damage_type": dmg_type,
            "damage_confidence": dmg_conf,
            "smart_score": smart_score,
        })

    damaged = [p for p in parts if p["damage_type"] != "none"]
    best = max(damaged, key=lambda p: p["smart_score"]) if damaged else None

    # Pipeline compatibility: build_repair_plan_from_detection and server expect detected_pairs
    detected_pairs = [
        {
            "part": p["part"],
            "damage_type": p["damage_type"],
            "part_confidence": p["part_confidence"],
            "damage_confidence": p["damage_confidence"],
            "smart_score": p["smart_score"],
        }
        for p in parts
        if p["damage_type"] != "none"
    ]

    return {
        "image_path": image_path,
        "parts": parts,
        "best_damage": best,
        "detected_pairs": detected_pairs,
    }


# Legacy name for callers that use detect()
def detect(image_path, weights=None, threshold=0.35, **kwargs):
    """Legacy entrypoint: same as detect_damage with part_thresh=threshold."""
    return detect_damage(image_path, part_thresh=threshold, **kwargs)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--part_thresh", "--threshold", type=float, default=0.35, dest="part_thresh")
    ap.add_argument("--parts_weights", default="./models/damage_detection/parts_detector_ssd.pth")
    ap.add_argument("--damage_weights", default="./models/damage_detection/damage_classifier_resnet18.pth")
    ap.add_argument("--debug", action="store_true", help="Ignored; kept for CLI compatibility")
    args = ap.parse_args()
    result = detect_damage(
        args.image,
        parts_weights=args.parts_weights,
        damage_weights=args.damage_weights,
        part_thresh=args.part_thresh,
    )
    print(json.dumps(result, indent=2))
