import torch
import json
from PIL import Image
import numpy as np
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision import transforms, ops

PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES


def build_model_for_inference(weights_path, device):
    """Build model matching the training architecture and load weights."""
    num_classes = len(CLASSES)
    
    # Build model with the same architecture as training (no pretrained weights)
    model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
    
    # Load the trained weights
    state = torch.load(weights_path, map_location=device)
    
    try:
        # Try strict loading first
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try with strict=False and handle mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        
        if missing_keys:
            print(f"[WARNING] Missing keys in checkpoint: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)}")
    
    model.to(device)
    model.eval()
    return model


def detect(image_path, weights="./models/damage_detection/mobilenet_ssd.pth",
           threshold=0.15, debug=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_for_inference(weights, device)

    tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img_t = tf(img).to(device)

    with torch.no_grad():
        out = model([img_t])[0]

    # NMS: more forgiving threshold to keep nearby boxes
    keep_indices = ops.nms(out["boxes"], out["scores"], 0.5)

    boxes = out["boxes"][keep_indices].detach().cpu().numpy()
    scores = out["scores"][keep_indices].detach().cpu().numpy()
    labels = out["labels"][keep_indices].detach().cpu().numpy()

    parts = []
    damages = []

    # Use separate thresholds for parts and damages
    # Damages naturally have lower confidence in SSD, especially when model is undertrained
    # For undertrained models, be very permissive with damages
    part_threshold = max(0.05, float(threshold))
    damage_threshold = 0.003  # Very low threshold for damages since model is undertrained

    for box, score, label in zip(boxes, scores, labels):
        cls_name = CLASSES[int(label)]
        
        if cls_name in PARTS:
            if score >= part_threshold:
                parts.append((cls_name, box, float(score)))
        elif cls_name in DAMAGES:
            if score >= damage_threshold:
                damages.append((cls_name, box, float(score)))

    if debug:
        print(f"[DEBUG] total kept boxes: {len(boxes)}")
        for cls_name, box, sc in parts:
            print(f"[DEBUG] PART  {cls_name:16s} score={sc:.3f} box={box}")
        for cls_name, box, sc in damages:
            print(f"[DEBUG] DAMAGE {cls_name:16s} score={sc:.3f} box={box}")

    # Pair logic: containment of damage inside part, but relaxed
    detected_pairs = []
    used_damages = set()

    for p_name, p_box, p_conf in parts:
        best_dmg = None
        best_coverage = 0.0
        best_dmg_idx = -1

        px1, py1, px2, py2 = p_box

        for i, (d_name, d_box, d_conf) in enumerate(damages):
            if i in used_damages:
                continue

            dx1, dy1, dx2, dy2 = d_box

            x1 = max(px1, dx1)
            y1 = max(py1, dy1)
            x2 = min(px2, dx2)
            y2 = min(py2, dy2)

            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            intersection_area = inter_w * inter_h

            if intersection_area <= 0:
                continue

            d_area = max(1e-6, (dx2 - dx1) * (dy2 - dy1))
            coverage = intersection_area / d_area

            # Relaxed coverage threshold so imperfect boxes still pair
            if coverage > 0.15 and coverage > best_coverage:
                best_coverage = coverage
                best_dmg = (d_name, d_conf)
                best_dmg_idx = i

        if best_dmg is not None:
            dmg_name, dmg_conf = best_dmg
            detected_pairs.append({
                "part": p_name,
                "part_confidence": float(p_conf),
                "damage_type": dmg_name,
                "damage_confidence": float(dmg_conf),
                "overlap_iou": float(best_coverage)
            })
            used_damages.add(best_dmg_idx)

    # Fallback: if we saw damage but no pair passed coverage threshold,
    # assign each damage to the nearest part by center distance.
    if not detected_pairs and damages and parts:
        if debug:
            print("[DEBUG] No pairs from coverage; using nearest-part fallback.")
        for d_name, d_box, d_conf in damages:
            dx = 0.5 * (d_box[0] + d_box[2])
            dy = 0.5 * (d_box[1] + d_box[3])

            best_p = None
            best_dist = 1e9
            for p_name, p_box, p_conf in parts:
                px = 0.5 * (p_box[0] + p_box[2])
                py = 0.5 * (p_box[1] + p_box[3])
                dist = abs(px - dx) + abs(py - dy)
                if dist < best_dist:
                    best_dist = dist
                    best_p = (p_name, p_conf)

            if best_p is not None:
                p_name, p_conf = best_p
                detected_pairs.append({
                    "part": p_name,
                    "part_confidence": float(p_conf),
                    "damage_type": d_name,
                    "damage_confidence": float(d_conf),
                    "overlap_iou": 0.0
                })

    if debug:
        print(f"[DEBUG] detected_pairs: {json.dumps(detected_pairs, indent=2)}")

    return {
        "image_path": image_path,
        "detected_pairs": detected_pairs
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="./models/damage_detection/mobilenet_ssd.pth")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    result = detect(
        args.image,
        weights=args.weights,
        threshold=args.threshold,
        debug=args.debug,
    )
    print(json.dumps(result, indent=2))
