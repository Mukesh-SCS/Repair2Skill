"""
================================================================================
DESCRIPTION:
    Predict damage types and chair parts with a MobileNetV3 classifier.
    Returns Stage-I JSON with filtered top part↔damage pairs for planning.

USAGE:
    from scripts.detect_damage import detect_damage_and_parts
    report = detect_damage_and_parts("chair.jpg", "./models/damage_detection/part_detector.pth")

OUTPUTS:
    {
      "detected_damages": [{"type": "...", "confidence": float}],
      "detected_parts":   [{"part": "...", "confidence": float}],
      "detected_pairs":   [{"part": "...", "damage_type": "...",
                            "damage_confidence": float, "part_confidence": float}]
    }

ARGUMENTS:
    image_path: str
    model_path: str
Author Info: Mukesh Mani Tripathi
================================================================================
"""
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from scripts.train_part_detector import FurnitureRepairModel

PART_CLASSES = [
    "seat","back","front_left_leg","front_right_leg",
    "back_left_leg","back_right_leg","armrest_left","armrest_right"
]
DAMAGE_CLASSES = ["missing","cracked","broken","loose","scratched"]

def _load_model(model_path: str):
    model = FurnitureRepairModel()
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

def detect_damage_and_parts(image_path: str, model_path: str,
                            thresh_damage: float = 0.60,
                            thresh_part: float = 0.60,
                            top_k_parts: int = 1):
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)

    model = _load_model(model_path)
    with torch.no_grad():
        damage_logits, part_logits = model(x)
        damage_probs = torch.sigmoid(damage_logits).squeeze().tolist()
        part_probs   = torch.sigmoid(part_logits).squeeze().tolist()

    detected_damages = [
        {"type": DAMAGE_CLASSES[i], "confidence": float(damage_probs[i])}
        for i in range(len(DAMAGE_CLASSES)) if damage_probs[i] >= thresh_damage
    ]
    detected_parts = [
        {"part": PART_CLASSES[i], "confidence": float(part_probs[i])}
        for i in range(len(PART_CLASSES)) if part_probs[i] >= thresh_part
    ]

    # Pair the single best damage with top-K parts (keeps output tight and correct)
    pairs = []
    if detected_damages and detected_parts:
        best_damage = max(detected_damages, key=lambda d: d["confidence"])
        top_parts = sorted(detected_parts, key=lambda p: p["confidence"], reverse=True)[:top_k_parts]
        for p in top_parts:
            pairs.append({
                "part": p["part"],
                "damage_type": best_damage["type"],
                "damage_confidence": best_damage["confidence"],
                "part_confidence": p["confidence"],
            })

    return {
        "detected_damages": detected_damages,
        "detected_parts": detected_parts,
        "detected_pairs": pairs
    }
