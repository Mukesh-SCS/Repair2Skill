import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from scripts.train_part_detector import FurnitureRepairModel
import json
import os

PART_CLASSES = [
    "seat", "back", "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
]

DAMAGE_CLASSES = ["missing", "cracked", "broken", "loose", "scratched"]

class DamagePartClassifier(nn.Module):
    def __init__(self, num_damage_classes=5, num_part_classes=8):
        super(DamagePartClassifier, self).__init__()
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.damage_classifier = nn.Sequential(
            nn.Linear(576, 128), nn.ReLU(), nn.Linear(128, num_damage_classes)
        )
        self.part_classifier = nn.Sequential(
            nn.Linear(576, 128), nn.ReLU(), nn.Linear(128, num_part_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.damage_classifier(x), self.part_classifier(x)

def load_model(model_path):
    model = FurnitureRepairModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def detect_damage_and_parts(image_path, model_path):
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

    detected_damages = [
        {"type": d, "confidence": damage_probs[i]}
        for i, d in enumerate(DAMAGE_CLASSES) if damage_probs[i] > 0.5
    ]

    detected_parts = [
        {"part": p, "confidence": part_probs[i]}
        for i, p in enumerate(PART_CLASSES) if part_probs[i] > 0.5
    ]

    return {
        "detected_damages": detected_damages,
        "detected_parts": detected_parts
    }