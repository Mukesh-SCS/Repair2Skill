import os
import json
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class FurnitureDetectionDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transforms=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.images_dir = images_dir
        self.transforms = transforms

        self.class_names = [
            "__background__", "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right",
            "missing", "cracked", "broken", "loose", "scratched"
        ]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.images_dir, ann["filename"])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        parts = ann.get("parts", {})
        for damage in ann.get("damages", []):
            part = damage["part"]
            label = damage["type"]
            if part in parts:
                box = parts[part]  # [x1, y1, x2, y2]
                boxes.append(box)
                labels.append(self.name_to_idx[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_transform():
    return T.Compose([
        T.ToTensor()
    ])


def train():
    dataset = FurnitureDetectionDataset(
        annotations_file="./data/synthetic_damage/annotations.json",
        images_dir="./data/synthetic_damage/images",
        transforms=get_transform()
    )

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 14  # background + 13 labels
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    model.train()
    for epoch in range(10):
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {losses.item():.4f}")

    os.makedirs("./models/damage_detection", exist_ok=True)
    torch.save(model.state_dict(), "./models/damage_detection/frcnn_model.pth")
    print("Faster R-CNN model saved.")

if __name__ == "__main__":
    train()
