import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from scripts.train_detector_frcnn import FurnitureDetectionDataset


def detect_parts(image_path, model_path):
    """Detect parts and damage in furniture image with bounding boxes"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 14  # From model_config.yaml
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Process outputs
    class_names = ["__background__", "seat", "back", "front_left_leg", "front_right_leg",
                   "back_left_leg", "back_right_leg", "armrest_left", "armrest_right",
                   "missing", "cracked", "broken", "loose", "scratched"]
    
    results = {
        "detected_damages": [],
        "detected_parts": []
    }
    
    # Threshold for detection
    threshold = 0.5
    
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > threshold:
            class_name = class_names[label]
            if class_name in ["seat", "back", "front_left_leg", "front_right_leg",
                             "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"]:
                results["detected_parts"].append({
                    "part": class_name,
                    "confidence": float(score),
                    "bbox": box.cpu().numpy().tolist()
                })
            else:
                results["detected_damages"].append({
                    "type": class_name,
                    "confidence": float(score),
                    "bbox": box.cpu().numpy().tolist()
                })
    
    return results

def detect_damage_from_image(image_path):
    """Main function to detect damage from image"""
    model_path = "./models/damage_detection/frcnn_model.pth"
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return {"error": "Model not found"}
    
    try:
        results = detect_parts(image_path, model_path)
        return results
    except Exception as e:
        print(f"Error during detection: {e}")
        return {"error": str(e)}