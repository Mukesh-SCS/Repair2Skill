import torch
import cv2
import json
import os
import numpy as np
from PIL import Image

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PARTS = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGES = ["missing", "cracked", "broken", "loose", "scratched"]

CLASSES = ["__background__"] + PARTS + DAMAGES


def load_model(weights_path, device):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def visualize_predictions(image_path, weights_path, out_path="viz_output.jpg", threshold=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(weights_path, device)

    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        preds = model([img_tensor])[0]

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    img_draw = img.copy()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        cls_name = CLASSES[label]
        color = (0, 255, 0) if cls_name in PARTS else (0, 0, 255)

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_draw, f"{cls_name} {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

    cv2.imwrite(out_path, img_draw)
    print(f"[OK] Saved visualization: {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="./models/damage_detection/frcnn_model.pth")
    ap.add_argument("--out", default="viz_output.jpg")
    args = ap.parse_args()

    visualize_predictions(args.image, args.weights, args.out)
