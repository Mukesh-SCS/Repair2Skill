import torch
import json
from PIL import Image
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
from torchvision import transforms

PARTS = [
    "seat","back","front_left_leg","front_right_leg",
    "back_left_leg","back_right_leg","armrest_left","armrest_right"
]

DAMAGES = ["missing","cracked","broken","loose","scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES


def load_model(weights_path):
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    num_classes = len(CLASSES)

    # Get the in_channels for each feature level
    in_channels = [module[1].in_channels for module in model.head.classification_head.module_list]
    
    # Set num_anchors (6 per location for SSDLite)
    num_anchors = [6] * len(in_channels)
    
    # Use GroupNorm instead of BatchNorm to avoid issues with small batch sizes
    norm_layer = lambda num_channels: torch.nn.GroupNorm(min(32, max(1, num_channels // 4)), num_channels)
    
    # Replace the heads with ones configured for our num_classes
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer
    )
    model.head.regression_head = SSDLiteRegressionHead(
        in_channels, num_anchors, norm_layer=norm_layer
    )

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def detect(image_path, weights="./models/damage_detection/mobilenet_ssd.pth", threshold=0.25):
    model = load_model(weights)
    tf = transforms.Compose([transforms.Resize((320,320)), transforms.ToTensor()])

    img = Image.open(image_path).convert("RGB")
    img_t = tf(img)

    with torch.no_grad():
        out = model([img_t])[0]

    boxes = out["boxes"].numpy()
    scores = out["scores"].numpy()
    labels = out["labels"].numpy()

    parts = []
    damages = []

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        cls = CLASSES[label]
        if cls in PARTS:
            parts.append((cls, box, score))
        elif cls in DAMAGES:
            damages.append((cls, box, score))

    # Pair part + damage by highest overlap
    detected_pairs = []
    for p_name, p_box, p_conf in parts:
        best_dmg = None
        best_iou = 0

        for d_name, d_box, d_conf in damages:
            # simple overlap test
            x1 = max(p_box[0], d_box[0])
            y1 = max(p_box[1], d_box[1])
            x2 = min(p_box[2], d_box[2])
            y2 = min(p_box[3], d_box[3])
            inter = max(0, x2-x1)*max(0,y2-y1)
            if inter > best_iou:
                best_iou = inter
                best_dmg = (d_name, d_conf)

        if best_dmg:
            dmg_name, dmg_conf = best_dmg
            detected_pairs.append({
                "part": p_name,
                "part_confidence": float(p_conf),
                "damage_type": dmg_name,
                "damage_confidence": float(dmg_conf)
            })

    return {
        "image_path": image_path,
        "detected_pairs": detected_pairs
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    result = detect(args.image)
    print(json.dumps(result, indent=2))
