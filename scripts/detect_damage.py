import torch
import json
import sys
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
    
    # Check if state dict is from WeightedLossModel wrapper
    if isinstance(state, dict):
        # If it's a checkpoint dict with 'model_state' key
        if 'model_state' in state:
            state_dict = state['model_state']
        else:
            state_dict = state
        
        # Remove 'model.' prefix if present (from WeightedLossModel wrapper)
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    else:
        state_dict = state
    
    try:
        # Try strict loading first
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try with strict=False and handle mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"[WARNING] Missing keys in checkpoint: {len(missing_keys)}", file=sys.stderr)
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)}", file=sys.stderr)
    
    model.to(device)
    model.eval()
    return model


def detect(image_path, weights="./models/damage_detection/mobilenet_ssd.pth",
           threshold=0.15, debug=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_for_inference(weights, device)

    # CRITICAL: Preprocessing must EXACTLY match training!
    # Training uses: resize with aspect ratio + center padding + normalize
    # We must do the same here for consistent predictions
    
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    target_size = 320
    
    # Step 1: Resize maintaining aspect ratio (same as training)
    img_resized = img.copy()
    img_resized.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    new_w, new_h = img_resized.size
    
    # Step 2: Pad to square with gray background (same as training)
    img_padded = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))
    
    # Calculate inverse transform for bbox mapping back to original coords
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    # Step 3: Convert to tensor and normalize (same as training)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_t = tf(img_padded).to(device)

    with torch.no_grad():
        out = model([img_t])[0]

    # NMS: more forgiving threshold to keep nearby boxes
    keep_indices = ops.nms(out["boxes"], out["scores"], 0.5)

    boxes = out["boxes"][keep_indices].detach().cpu().numpy()
    scores = out["scores"][keep_indices].detach().cpu().numpy()
    labels = out["labels"][keep_indices].detach().cpu().numpy()
    
    # Transform boxes back to original image coordinates
    # Reverse: subtract padding, then divide by scale
    boxes_orig = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Remove padding offset
        x1 = (x1 - paste_x) / scale_x
        y1 = (y1 - paste_y) / scale_y
        x2 = (x2 - paste_x) / scale_x
        y2 = (y2 - paste_y) / scale_y
        # Clamp to original image bounds
        x1 = max(0, min(orig_w, x1))
        x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h, y1))
        y2 = max(0, min(orig_h, y2))
        boxes_orig.append([x1, y1, x2, y2])
    
    boxes = np.array(boxes_orig) if boxes_orig else boxes

    parts = []
    damages = []

    # Use separate thresholds for parts and damages
    # Lower minimum thresholds to allow more detections, especially for undertrained models
    part_threshold = max(0.05, float(threshold))  # Lowered from 0.10 to allow more detections
    damage_threshold = max(0.03, float(threshold) * 0.6)  # Lowered from 0.08, now 60% of part threshold

    for box, score, label in zip(boxes, scores, labels):
        cls_name = CLASSES[int(label)]
        
        if cls_name in PARTS:
            if score >= part_threshold:
                parts.append((cls_name, box, float(score)))
        elif cls_name in DAMAGES:
            if score >= damage_threshold:
                damages.append((cls_name, box, float(score)))

    if debug:
        print(f"[DEBUG] total kept boxes: {len(boxes)}", file=sys.stderr)
        for cls_name, box, sc in parts:
            print(f"[DEBUG] PART  {cls_name:16s} score={sc:.3f} box={box}", file=sys.stderr)
        for cls_name, box, sc in damages:
            print(f"[DEBUG] DAMAGE {cls_name:16s} score={sc:.3f} box={box}", file=sys.stderr)

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
            # Lowered from 0.15 to 0.10 to allow more pairings
            if coverage > 0.10 and coverage > best_coverage:
                best_coverage = coverage
                best_dmg = (d_name, d_conf)
                best_dmg_idx = i

        if best_dmg is not None:
            dmg_name, dmg_conf = best_dmg
            # Calculate smart score for better pair selection
            overlap_bonus = max(0.5, best_coverage) if best_coverage > 0.15 else 0.3
            smart_score = float(p_conf) * float(dmg_conf) * overlap_bonus
            
            detected_pairs.append({
                "part": p_name,
                "part_confidence": float(p_conf),
                "damage_type": dmg_name,
                "damage_confidence": float(dmg_conf),
                "overlap_iou": float(best_coverage),
                "smart_score": smart_score  # Add smart score for better selection
            })
            used_damages.add(best_dmg_idx)

    # Fallback: if we saw damage but no pair passed coverage threshold,
    # assign each damage to the nearest part by center distance.
    if not detected_pairs and damages and parts:
        if debug:
            print("[DEBUG] No pairs from coverage; using nearest-part fallback.", file=sys.stderr)
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
                # Calculate smart score for fallback pairs too
                smart_score = float(p_conf) * float(d_conf) * 0.3  # Lower bonus for fallback
                detected_pairs.append({
                    "part": p_name,
                    "part_confidence": float(p_conf),
                    "damage_type": d_name,
                    "damage_confidence": float(d_conf),
                    "overlap_iou": 0.0,
                    "smart_score": smart_score
                })

    if debug:
        print(f"[DEBUG] detected_pairs: {json.dumps(detected_pairs, indent=2)}", file=sys.stderr)

    # If no pairs detected but we have parts, it means the model isn't detecting damages
    if not detected_pairs and parts:
        print("[WARN] Model detected parts but no damages.", file=sys.stderr)
        if max([d[2] for d in damages], default=0.0) < 0.1:
            print("[HINT] Model appears undertrained. Retrain with: python scripts/train_detector_mobilenet.py --epochs 100", file=sys.stderr)
        else:
            print("[HINT] Try lowering --threshold parameter (current: {:.3f})".format(threshold), file=sys.stderr)
    
    return {
        "image_path": image_path,
        "detected_pairs": detected_pairs
    }



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="*", required=True, help="Image file(s) to process")
    ap.add_argument("--weights", default="./models/damage_detection/mobilenet_ssd.pth")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--visualize", action="store_true", help="Save detection visualization")
    args = ap.parse_args()

    results = []
    for img_path in args.image:
        result = detect(
            img_path,
            weights=args.weights,
            threshold=args.threshold,
            debug=args.debug,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

        if args.visualize:
            try:
                img = Image.open(img_path).convert("RGB")
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                for pair in result.get("detected_pairs", []):
                    box = pair.get("part_confidence", None)
                    dmg_box = pair.get("damage_confidence", None)
                    part = pair.get("part", "")
                    dmg_type = pair.get("damage_type", "")
                    # Draw part box (blue)
                    if "part" in pair and "overlap_iou" in pair:
                        color = "blue"
                        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        ax.text(box[0], box[1], part, color=color, fontsize=8)
                    # Draw damage box (red)
                    if "damage_type" in pair and "overlap_iou" in pair:
                        color = "red"
                        rect = patches.Rectangle((dmg_box[0], dmg_box[1]), dmg_box[2]-dmg_box[0], dmg_box[3]-dmg_box[1], linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        ax.text(dmg_box[0], dmg_box[1], dmg_type, color=color, fontsize=8)
                plt.axis('off')
                out_path = img_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"[VISUALIZE] Saved detection visualization to {out_path}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Visualization failed: {e}", file=sys.stderr)
