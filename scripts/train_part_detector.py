# ==================== scripts/train_part_detector.py ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small  # Add this import
import json
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class FurnitureDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.images_dir = images_dir
        self.transform = transform
        
        # Create label mapping
        self.damage_types = ["missing", "cracked", "broken", "loose", "scratched"]
        self.part_types = [
            "seat", "back", "front_left_leg", "front_right_leg", 
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
        ]
        
        self.damage_to_idx = {damage: idx for idx, damage in enumerate(self.damage_types)}
        self.part_to_idx = {part: idx for idx, part in enumerate(self.part_types)}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, annotation['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create labels
        damage_labels = torch.zeros(len(self.damage_types))
        part_labels = torch.zeros(len(self.part_types))
        
        for damage in annotation['damages']:
            damage_idx = self.damage_to_idx[damage['type']]
            part_idx = self.part_to_idx[damage['part']]
            
            damage_labels[damage_idx] = 1
            part_labels[part_idx] = 1
        
        return image, damage_labels, part_labels

class FurnitureRepairModel(nn.Module):
    def __init__(self, num_damage_classes=5, num_part_classes=8):
        super(FurnitureRepairModel, self).__init__()
        
        # Use MobileNetV3-Small as backbone
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove final classification layer
        
        # Add custom heads (input features = 1024 for mobilenet_v3_small)
        self.damage_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_damage_classes),
            nn.Sigmoid()
        )
        
        self.part_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_part_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        damage_output = self.damage_classifier(features)
        part_output = self.part_classifier(features)
        return damage_output, part_output

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = FurnitureDataset(
        annotations_file="./data/synthetic_damage/annotations.json",
        images_dir="./data/synthetic_damage/images",
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = FurnitureRepairModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, damage_labels, part_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            damage_labels = damage_labels.to(device)
            part_labels = part_labels.to(device)
            
            optimizer.zero_grad()
            
            damage_outputs, part_outputs = model(images)
            
            damage_loss = criterion(damage_outputs, damage_labels)
            part_loss = criterion(part_outputs, part_labels)
            
            total_loss = damage_loss + part_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, damage_labels, part_labels in val_loader:
                images = images.to(device)
                damage_labels = damage_labels.to(device)
                part_labels = part_labels.to(device)

                damage_outputs, part_outputs = model(images)

                damage_loss = criterion(damage_outputs, damage_labels)
                part_loss = criterion(part_outputs, part_labels)

                total_loss = damage_loss + part_loss
                val_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step()
    
    # Save model
    os.makedirs("./models/damage_detection", exist_ok=True)
    torch.save(model.state_dict(), "./models/damage_detection/part_detector.pth")
    print("Model saved successfully!")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("./outputs/training_curves.png")
    plt.show()

if __name__ == "__main__":
    train_model()