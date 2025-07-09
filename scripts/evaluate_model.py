# ==================== scripts/evaluate_model.py ====================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scripts.train_part_detector import FurnitureDataset, FurnitureRepairModel
import torchvision.transforms as transforms

def evaluate_model(model_path, test_data_path):
    """Evaluate the trained model on test data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = FurnitureRepairModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FurnitureDataset(
        annotations_file=f"{test_data_path}/annotations.json",
        images_dir=f"{test_data_path}/images",
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluation metrics
    all_damage_preds = []
    all_damage_labels = []
    all_part_preds = []
    all_part_labels = []
    
    total_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for images, damage_labels, part_labels, _ in test_loader:
            images = images.to(device)
            damage_labels = damage_labels.to(device)
            part_labels = part_labels.to(device)
            
            damage_outputs, part_outputs = model(images)
            
            # Calculate loss
            damage_loss = criterion(damage_outputs, damage_labels)
            part_loss = criterion(part_outputs, part_labels)
            total_loss += (damage_loss + part_loss).item()
            
            # Convert to predictions
            damage_preds = (damage_outputs > 0.5).float()
            part_preds = (part_outputs > 0.5).float()
            
            all_damage_preds.extend(damage_preds.cpu().numpy())
            all_damage_labels.extend(damage_labels.cpu().numpy())
            all_part_preds.extend(part_preds.cpu().numpy())
            all_part_labels.extend(part_labels.cpu().numpy())
    
    # Calculate metrics
    all_damage_preds = np.array(all_damage_preds)
    all_damage_labels = np.array(all_damage_labels)
    all_part_preds = np.array(all_part_preds)
    all_part_labels = np.array(all_part_labels)
    
    # Damage classification report
    damage_types = ["missing", "cracked", "broken", "loose", "scratched"]
    part_types = ["seat", "back", "front_left_leg", "front_right_leg", 
                  "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"]
    
    print("=== DAMAGE DETECTION EVALUATION ===")
    for i, damage_type in enumerate(damage_types):
        print(f"\n{damage_type.upper()} Detection:")
        print(classification_report(all_damage_labels[:, i], all_damage_preds[:, i]))
    
    print("\n=== PART DETECTION EVALUATION ===")
    for i, part_type in enumerate(part_types):
        print(f"\n{part_type.upper()} Detection:")
        print(classification_report(all_part_labels[:, i], all_part_preds[:, i]))
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Damage confusion matrices
    for i in range(min(5, len(damage_types))):
        if i < 3:
            ax = axes[0, i]
            cm = confusion_matrix(all_damage_labels[:, i], all_damage_preds[:, i])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f'{damage_types[i]} Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
    
    # Part confusion matrices (top 3)
    for i in range(min(3, len(part_types))):
        ax = axes[1, i]
        cm = confusion_matrix(all_part_labels[:, i], all_part_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Greens')
        ax.set_title(f'{part_types[i]} Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('./outputs/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save evaluation results
    results = {
        "total_loss": total_loss / len(test_loader),
        "damage_accuracy": np.mean(all_damage_preds == all_damage_labels),
        "part_accuracy": np.mean(all_part_preds == all_part_labels),
        "damage_types": damage_types,
        "part_types": part_types
    }
    
    with open('./outputs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOverall Test Loss: {results['total_loss']:.4f}")
    print(f"Damage Detection Accuracy: {results['damage_accuracy']:.4f}")
    print(f"Part Detection Accuracy: {results['part_accuracy']:.4f}")