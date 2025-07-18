import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from scripts.train_part_detector import FurnitureRepairModel
import os

PART_CLASSES = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGE_CLASSES = ["missing", "cracked", "broken", "loose", "scratched"]

def load_model(model_path):
    model = FurnitureRepairModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def detect_damage_and_parts(image_path, model_path, damage_threshold=0.5, part_threshold=0.5):
    if not os.path.exists(model_path):
        return {"error": "Model not found"}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    model = load_model(model_path)
    with torch.no_grad():
        damage_logits, part_logits = model(input_tensor)
        damage_probs = torch.sigmoid(damage_logits).squeeze().tolist()
        part_probs = torch.sigmoid(part_logits).squeeze().tolist()

    detected_pairs = []
    for i, d_conf in enumerate(damage_probs):
        if d_conf > damage_threshold:
            for j, p_conf in enumerate(part_probs):
                if p_conf > part_threshold:
                    detected_pairs.append({
                        "part": PART_CLASSES[j],
                        "damage_type": DAMAGE_CLASSES[i],
                        "damage_confidence": d_conf,
                        "part_confidence": p_conf
                    })

    return {
        "detected_pairs": detected_pairs
    }

def filter_detected_pairs(detected_pairs, damage_threshold=0.7, part_threshold=0.7):
    # Filter pairs by thresholds
    filtered = [p for p in detected_pairs if p['damage_confidence'] >= damage_threshold and p['part_confidence'] >= part_threshold]

    # Keep only highest damage confidence per part
    best_pairs = {}
    for p in filtered:
        part = p['part']
        if part not in best_pairs or p['damage_confidence'] > best_pairs[part]['damage_confidence']:
            best_pairs[part] = p

    return list(best_pairs.values())

if __name__ == "__main__":
    # For quick test
    result = detect_damage_and_parts("example.jpg", "./models/damage_detection/part_detector.pth")
    print(result)
