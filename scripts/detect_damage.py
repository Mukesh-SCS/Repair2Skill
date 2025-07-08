import torch
import torchvision.transforms as T
from PIL import Image
from utils.openai_utils import analyze_image # Assuming analyze_image is a function that returns a damage report


def detect_damage_from_image(image_path):
    report = analyze_image(image_path)
    return report

def detect_parts(image_path, model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    detected_parts = predictions['labels'].numpy()
    # Compare detected_parts with expected parts to detect missing parts
    print(f"Detected parts: {detected_parts}")

if __name__ == "__main__":
    detect_parts("../data/synthetic_damage/chair_missing_leg.png",
                 "../models/damage_detection/part_detector.pth")



