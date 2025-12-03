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

    # Debug: show ALL detections before any filtering
    if debug:
        print(f"[DEBUG] ============ ALL RAW DETECTIONS ({len(boxes)} total) ============")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            cls_name = CLASSES[int(label)]
            print(f"[DEBUG] [{i:2d}] {cls_name:16s} score={score:.4f}")

    parts = []
    damages = []

    # Use separate thresholds for parts and damages
    # Parts: use the provided threshold
    # Damages: use very low threshold since even well-trained models
    #          produce low confidence for damage types. We rely on
    #          pairing logic, spatial overlap, and smart scoring.
    # NOTE: If you see false positives, increase damage_threshold to 0.002-0.003
    part_threshold = max(0.05, float(threshold))
    damage_threshold = 0.001  # Very permissive - filtering happens via scoring

    for box, score, label in zip(boxes, scores, labels):
        cls_name = CLASSES[int(label)]
        
        if cls_name in PARTS:
            if score >= part_threshold:
                parts.append((cls_name, box, float(score)))
        elif cls_name in DAMAGES:
            if score >= damage_threshold:
                damages.append((cls_name, box, float(score)))

    if debug:
        print(f"[DEBUG] total boxes before threshold: {len(boxes)}")
        print(f"[DEBUG] part_threshold={part_threshold:.4f}, damage_threshold={damage_threshold:.4f}")
        print(f"[DEBUG] Parts detected: {len(parts)}, Damages detected: {len(damages)}")
        for cls_name, box, sc in parts:
            print(f"[DEBUG] PART  {cls_name:16s} score={sc:.3f} box={[f'{b:.2f}' for b in box]}")
        for cls_name, box, sc in damages:
            print(f"[DEBUG] DAMAGE {cls_name:16s} score={sc:.3f} box={[f'{b:.2f}' for b in box]}")

    # Pair logic: containment of damage inside part, but relaxed
    detected_pairs = []
    used_damages = set()

    if debug:
        print(f"[DEBUG] Pairing logic starting with {len(parts)} parts and {len(damages)} damages")

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
                if debug:
                    print(f"[DEBUG]   Part {p_name} ← Damage {d_name} (coverage={coverage:.3f}, damage_conf={d_conf:.3f})")

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
            if debug:
                print(f"[DEBUG]   → Paired {p_name} with {dmg_name}")

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

    # Re-score using ONLY damage confidence (not part confidence) to favor accuracy over confidence
    # This reduces false positives where a high-confidence part gets paired with low-confidence damage
    if detected_pairs and debug:
        print("\n[DEBUG] ===== SCORING ANALYSIS =====")
        
    for pair in detected_pairs:
        dmg_conf = pair['damage_confidence']
        overlap = pair['overlap_iou']
        part_conf = pair['part_confidence']
        
        # Smart score: favor spatial overlap and damage confidence over part confidence
        # overlap_bonus: rewards good spatial alignment
        overlap_bonus = max(0.5, overlap) if overlap > 0.15 else 0.3
        
        # Score prioritizes: (1) damage confidence (most important for accuracy)
        #                    (2) overlap quality (spatial alignment)
        #                    (3) part confidence (trust high-conf parts less if damage is elsewhere)
        smart_score = dmg_conf * overlap_bonus * (0.5 + 0.5 * part_conf)
        
        pair["smart_score"] = float(smart_score)
        
        if debug:
            print(f"[DEBUG]   {pair['part']:15s} + {pair['damage_type']:10s}")
            print(f"[DEBUG]     dmg_conf={dmg_conf:.4f}, overlap={overlap:.3f}, part_conf={part_conf:.3f}")
            print(f"[DEBUG]     smart_score={smart_score:.6f}")
    
    if debug:
        print("[DEBUG] ==========================\n")

    if debug:
        print(f"[DEBUG] Final detected_pairs ({len(detected_pairs)} total):")
        for pair in detected_pairs:
            smart_score = pair.get('smart_score', 0)
            print(f"[DEBUG]   {pair['part']:15s} + {pair['damage_type']:10s} | smart_score={smart_score:.6f}")
        
        # Show which pair would be selected
        if detected_pairs:
            sorted_pairs = sorted(detected_pairs, key=lambda x: -x.get('smart_score', 0))
            best = sorted_pairs[0]
            print(f"[DEBUG] >>> TOP SELECTED: {best['part']} + {best['damage_type']} (score={best.get('smart_score', 0):.6f})")

    # Return results
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
