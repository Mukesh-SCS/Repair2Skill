# ==================== scripts/generate_synthetic_data.py ====================
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import random
import albumentations as A
from tqdm import tqdm

class SyntheticDataGenerator:
    def __init__(self, output_dir="./data/synthetic_damage/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        
        self.chair_parts = [
            "seat", "back", "front_left_leg", "front_right_leg", 
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
        ]
        
        self.damage_types = [
            "missing", "cracked", "broken", "loose", "scratched"
        ]
        
        # Data augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ])
    
    def generate_chair_template(self, width=640, height=480):
        """Generate a basic chair template"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Define chair parts with their bounding boxes
        parts = {
            "seat": [200, 200, 440, 250],
            "back": [220, 100, 420, 200],
            "front_left_leg": [200, 250, 230, 350],
            "front_right_leg": [410, 250, 440, 350],
            "back_left_leg": [220, 250, 250, 350],
            "back_right_leg": [390, 250, 420, 350],
            "armrest_left": [180, 150, 220, 180],
            "armrest_right": [420, 150, 460, 180]
        }
        
        # Draw chair parts
        for part, bbox in parts.items():
            draw.rectangle(bbox, outline='black', width=2, fill='lightgray')
        
        return img, parts
    
    def apply_damage(self, img, parts, damage_info):
        """Apply damage to specific parts"""
        draw = ImageDraw.Draw(img)
        
        for damage in damage_info:
            part = damage["part"]
            damage_type = damage["type"]
            
            if part in parts:
                bbox = parts[part]
                
                if damage_type == "missing":
                    # Remove part by drawing white rectangle
                    draw.rectangle(bbox, fill='white', outline='white')
                elif damage_type == "cracked":
                    # Draw crack lines
                    x1, y1, x2, y2 = bbox
                    for _ in range(random.randint(2, 5)):
                        start_x = random.randint(x1, x2)
                        start_y = random.randint(y1, y2)
                        end_x = random.randint(x1, x2)
                        end_y = random.randint(y1, y2)
                        draw.line([(start_x, start_y), (end_x, end_y)], 
                                fill='red', width=2)
                elif damage_type == "broken":
                    # Draw broken effect
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], fill='darkgray')
                    # Add some broken pieces
                    for _ in range(random.randint(3, 7)):
                        piece_x = random.randint(x1, x2)
                        piece_y = random.randint(y1, y2)
                        draw.ellipse([piece_x-5, piece_y-5, piece_x+5, piece_y+5], 
                                   fill='gray')
        
        return img
    
    def generate_dataset(self, num_samples=1000):
        """Generate synthetic dataset"""
        annotations = []
        
        for i in tqdm(range(num_samples), desc="Generating synthetic data"):
            # Generate base chair
            img, parts = self.generate_chair_template()
            
            # Generate random damage
            num_damages = random.randint(1, 3)
            damage_info = []
            damaged_parts = random.sample(self.chair_parts, num_damages)
            
            for part in damaged_parts:
                damage_type = random.choice(self.damage_types)
                damage_info.append({
                    "part": part,
                    "type": damage_type,
                    "severity": random.uniform(0.3, 1.0)
                })
            
            # Apply damage
            img = self.apply_damage(img, parts, damage_info)
            
            # Convert to numpy for augmentation
            img_np = np.array(img)
            
            # Apply augmentation
            augmented = self.augmentation(image=img_np)
            img_augmented = Image.fromarray(augmented['image'])
            
            # Save image
            img_filename = f"synthetic_{i:05d}.jpg"
            img_path = os.path.join(self.output_dir, "images", img_filename)
            img_augmented.save(img_path)
            
            # Create annotation
            annotation = {
                "image_id": i,
                "filename": img_filename,
                "width": img.width,
                "height": img.height,
                "damages": damage_info,
                "parts": parts
            }
            annotations.append(annotation)
        
        # Save annotations
        with open(os.path.join(self.output_dir, "annotations.json"), "w") as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Generated {num_samples} synthetic images with annotations")