import torch
import json
from PIL import Image
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
from torchvision import transforms, ops

PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES

def build_model_for_inference(weights_path, device):
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    num_classes = len(CLASSES)
    in_channels = [m[1].in_channels for m in model.head.classification_head.module_list]
    num_anchors = [6] * len(in_channels)
    norm_layer = lambda n: torch.nn.GroupNorm(min(32, max(1, n // 4)), n)
    
    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer=norm_layer)
    model.head.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer=norm_layer)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def detect(image_path, weights="./models/damage_detection/mobilenet_ssd.pth", threshold=0.15, debug=False):
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

    # Apply NMS (Non-Maximum Suppression) to remove duplicate overlapping boxes
    keep_indices = ops.nms(out["boxes"], out["scores"], 0.2) # Strict NMS
    
    boxes = out["boxes"][keep_indices].detach().cpu().numpy()
    scores = out["scores"][keep_indices].detach().cpu().numpy()
    labels = out["labels"][keep_indices].detach().cpu().numpy()

    parts = []
    damages = []

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold: continue
        
        cls_name = CLASSES[int(label)]
        if cls_name in PARTS:
            parts.append((cls_name, box, float(score)))
        elif cls_name in DAMAGES:
            damages.append((cls_name, box, float(score)))

    # Pair logic: Containment instead of IoU
    # We want to know if the DAMAGE is INSIDE the PART.
    detected_pairs = []
    
    used_damages = set()

    for p_name, p_box, p_conf in parts:
        best_dmg = None
        best_coverage = 0.0
        best_dmg_idx = -1

        for i, (d_name, d_box, d_conf) in enumerate(damages):
            if i in used_damages: continue

            # Intersection
            x1 = max(p_box[0], d_box[0])
            y1 = max(p_box[1], d_box[1])
            x2 = min(p_box[2], d_box[2])
            y2 = min(p_box[3], d_box[3])
            
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            intersection_area = inter_w * inter_h
            
            if intersection_area <= 0: continue

            # Damage Area
            d_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
            
            # Coverage: How much of the damage is inside this part?
            coverage = intersection_area / (d_area + 1e-6)

            # We accept matches if > 50% of the damage box is inside the part
            if coverage > 0.5 and coverage > best_coverage:
                best_coverage = coverage
                best_dmg = (d_name, d_conf)
                best_dmg_idx = i

        if best_dmg:
            dmg_name, dmg_conf = best_dmg
            detected_pairs.append({
                "part": p_name,
                "part_confidence": p_conf,
                "damage_type": dmg_name,
                "damage_confidence": dmg_conf,
                "overlap_iou": best_coverage # storing coverage here
            })
            used_damages.add(best_dmg_idx)

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
    args = ap.parse_args()
    print(json.dumps(detect(args.image, weights=args.weights, threshold=args.threshold), indent=2))