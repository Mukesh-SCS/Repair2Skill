import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import json
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class FurnitureDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.images_dir = images_dir
        self.transform = transform

        self.damage_types = ["missing", "cracked", "broken", "loose", "scratched"]
        self.part_types = [
            "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
        ]

        self.damage_to_idx = {d: i for i, d in enumerate(self.damage_types)}
        self.part_to_idx = {p: i for i, p in enumerate(self.part_types)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.images_dir, annotation['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        damage_labels = torch.zeros(len(self.damage_types))
        part_labels = torch.zeros(len(self.part_types))

        for damage in annotation['damages']:
            damage_labels[self.damage_to_idx[damage['type']]] = 1
            part_labels[self.part_to_idx[damage['part']]] = 1

        return image, damage_labels, part_labels

class FurnitureRepairModel(nn.Module):
    def __init__(self, num_damage_classes=5, num_part_classes=8):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights="IMAGENET1K_V1")  # updated for latest PyTorch
        self.backbone.classifier = nn.Identity()  # remove original classifier
        self.feature_dim = 576  # MobileNetV3 Small output size

        self.damage_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_damage_classes),
            nn.Sigmoid()
        )

        self.part_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_part_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.damage_classifier(features), self.part_classifier(features)


    def forward(self, x):
        features = self.backbone(x)
        return self.damage_classifier(features), self.part_classifier(features)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = FurnitureDataset(
        annotations_file="./data/synthetic_damage/annotations.json",
        images_dir="./data/synthetic_damage/images",
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = FurnitureRepairModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 20
    train_losses, val_losses = [], []
    start_time = time.time()

    os.makedirs("./models/damage_detection/checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for images, damage_labels, part_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, damage_labels, part_labels = images.to(device), damage_labels.to(device), part_labels.to(device)
            optimizer.zero_grad()
            damage_outputs, part_outputs = model(images)
            total_loss = criterion(damage_outputs, damage_labels) + criterion(part_outputs, part_labels)
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, damage_labels, part_labels in val_loader:
                images, damage_labels, part_labels = images.to(device), damage_labels.to(device), part_labels.to(device)
                damage_outputs, part_outputs = model(images)
                total_loss = criterion(damage_outputs, damage_labels) + criterion(part_outputs, part_labels)
                val_loss += total_loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Time: {time.time() - epoch_start:.2f}s")

        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"./models/damage_detection/checkpoints/epoch_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "./models/damage_detection/part_detector.pth")
    print(f"Training complete in {time.time() - start_time:.2f} seconds.")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("./outputs/training_curves.png")
    plt.close()

if __name__ == "__main__":
    train_model()
