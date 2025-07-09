# ==================== scripts/detect_damage.py ====================
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from scripts.train_part_detector import FurnitureRepairModel

def detect_parts(image_path, model_path):
    """Detect parts and damage in furniture image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = FurnitureRepairModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        damage_outputs, part_outputs = model(image_tensor)
    
    # Process outputs
    damage_types = ["missing", "cracked", "broken", "loose", "scratched"]
    part_types = [
        "seat", "back", "front_left_leg", "front_right_leg", 
        "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
    ]
    
    damage_probs = torch.sigmoid(damage_outputs).cpu().numpy()[0]
    part_probs = torch.sigmoid(part_outputs).cpu().numpy()[0]
    
    # Create results
    results = {
        "detected_damages": [],
        "detected_parts": []
    }
    
    # Threshold for detection
    threshold = 0.5
    
    for i, prob in enumerate(damage_probs):
        if prob > threshold:
            results["detected_damages"].append({
                "type": damage_types[i],
                "confidence": float(prob)
            })
    
    for i, prob in enumerate(part_probs):
        if prob > threshold:
            results["detected_parts"].append({
                "part": part_types[i],
                "confidence": float(prob)
            })
    
    return results

def detect_damage_from_image(image_path):
    """Main function to detect damage from image"""
    model_path = "./models/damage_detection/part_detector.pth"
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return {"error": "Model not found"}
    
    try:
        results = detect_parts(image_path, model_path)
        return results
    except Exception as e:
        print(f"Error during detection: {e}")
        return {"error": str(e)}