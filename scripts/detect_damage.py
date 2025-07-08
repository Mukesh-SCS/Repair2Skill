import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from utils.openai_utils import analyze_image  # Must be implemented in utils.openai_utils


def detect_damage_from_image(image_path):
    """
    Analyze the image and return a damage report.
    """
    report = analyze_image(image_path)
    return report


def detect_parts(image_path, model_path):
    """
    Detect parts in the image using a trained Faster R-CNN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    detected_parts = predictions['labels'].cpu().numpy()
    print(f"Detected parts: {detected_parts}")
    # Optionally return detected_parts for further use
    return detected_parts


if __name__ == "__main__":
    detect_parts("../data/synthetic_damage/chair_missing_leg.png",
                 "../models/damage_detection/part_detector.pth")



