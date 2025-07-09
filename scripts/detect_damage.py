import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from utils.openai_utils import analyze_image


def detect_damage_from_image(image_path):
    """
    Analyze the image and return a damage report.
    """
    report = analyze_image(image_path)
    return report


def detect_parts(image_path, model_path=None):
    """
    Detect parts in the image using a trained Faster R-CNN model.
    """
    # Skip PyTorch model loading
    result = analyze_image(image_path)
    print(result)
    return result


if __name__ == "__main__":
    detect_parts("../data/synthetic_damage/chair_missing_leg.png",
                 "../models/damage_detection/part_detector.pth")



