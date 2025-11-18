import torch
import json
from PIL import Image
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
from torchvision import transforms

PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES


def build_model_for_inference(weights_path, device):
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    num_classes = len(CLASSES)

    in_channels = [module[1].in_channels for module in model.head.classification_head.module_list]
    num_anchors = [6] * len(in_channels)

    norm_layer = lambda num_channels: torch.nn.GroupNorm(
        min(32, max(1, num_channels // 4)), num_channels
    )

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer
    )
    model.head.regression_head = SSDLiteRegressionHead(
        in_channels, num_anchors, norm_layer=norm_layer
    )

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def detect(image_path, weights="./models/damage_detection/mobilenet_ssd.pth",
           threshold=0.10, debug=False):
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

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()

    if debug:
        # Print raw top-k predictions for sanity
        order = np.argsort(-scores)
        print("[DEBUG] Top predictions (label, score):")
        for i in order[:20]:
            print(f"  {i}: {CLASSES[int(labels[i])]} @ {scores[i]:.3f}")

    parts = []
    damages = []

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        cls = CLASSES[int(label)]
        if cls in PARTS:
            parts.append((cls, box, float(score)))
        elif cls in DAMAGES:
            damages.append((cls, box, float(score)))

    # Pair part + damage by highest overlap
    detected_pairs = []
    for p_name, p_box, p_conf in parts:
        best_dmg = None
        best_iou = 0.0

        for d_name, d_box, d_conf in damages:
            x1 = max(p_box[0], d_box[0])
            y1 = max(p_box[1], d_box[1])
            x2 = min(p_box[2], d_box[2])
            y2 = min(p_box[3], d_box[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter <= 0:
                continue
            area_p = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            area_d = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
            union = area_p + area_d - inter
            iou = inter / max(union, 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_dmg = (d_name, d_conf)

        if best_dmg:
            dmg_name, dmg_conf = best_dmg
            detected_pairs.append({
                "part": p_name,
                "part_confidence": float(p_conf),
                "damage_type": dmg_name,
                "damage_confidence": float(dmg_conf),
                "overlap_iou": float(best_iou),
            })

    return {
        "image_path": image_path,
        "detected_pairs": detected_pairs
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="./models/damage_detection/mobilenet_ssd.pth")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    result = detect(args.image, weights=args.weights, threshold=args.threshold, debug=args.debug)
    print(json.dumps(result, indent=2))
