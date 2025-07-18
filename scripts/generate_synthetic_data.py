
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
            A.Blur(blur_limit=3, p=0.2),
        ])

    def generate_chair_template(self, width=640, height=480):
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

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

        for part, bbox in parts.items():
            draw.rectangle(bbox, outline='black', width=2, fill='lightgray')

        return img, parts

    def apply_damage(self, img, parts, damage_info):
        draw = ImageDraw.Draw(img)

        for damage in damage_info:
            part = damage["part"]
            damage_type = damage["type"]

            if part in parts:
                bbox = parts[part]
                if damage_type == "missing":
                    draw.rectangle(bbox, fill='white', outline='white')
                    parts.pop(part)  # Remove from annotation
                elif damage_type == "cracked":
                    x1, y1, x2, y2 = bbox
                    for _ in range(random.randint(2, 5)):
                        draw.line(
                            [(random.randint(x1, x2), random.randint(y1, y2)),
                             (random.randint(x1, x2), random.randint(y1, y2))],
                            fill='red', width=2)
                elif damage_type == "broken":
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], fill='darkgray')
                    for _ in range(random.randint(3, 7)):
                        px = random.randint(x1, x2)
                        py = random.randint(y1, y2)
                        draw.ellipse([px-5, py-5, px+5, py+5], fill='gray')
        return img

    def generate_dataset(self, num_samples=1000):
        annotations = []

        for i in tqdm(range(num_samples), desc="Generating synthetic data"):
            img, parts = self.generate_chair_template()

            num_damages = random.randint(1, 3)
            damage_info = []
            damaged_parts = random.sample(self.chair_parts, num_damages)

            for part in damaged_parts:
                damage_type = random.choice(self.damage_types)
                damage_info.append({
                    "part": part,
                    "type": damage_type,
                    "severity": round(random.uniform(0.3, 1.0), 2)
                })

            img = self.apply_damage(img, parts.copy(), damage_info)
            img_np = np.array(img)
            augmented = self.augmentation(image=img_np)
            img_augmented = Image.fromarray(augmented['image'])

            img_filename = f"synthetic_{i:05d}.jpg"
            img_path = os.path.join(self.output_dir, "images", img_filename)
            img_augmented.save(img_path)

            annotation = {
                "image_id": i,
                "filename": img_filename,
                "width": img_augmented.width,
                "height": img_augmented.height,
                "damages": damage_info,
                "parts": parts
            }
            annotations.append(annotation)

        with open(os.path.join(self.output_dir, "annotations.json"), "w") as f:
            json.dump(annotations, f, indent=2)

        print(f"Generated {num_samples} synthetic images with annotations")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic chair damage dataset")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic_damage/", help="Output directory")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(output_dir=args.output_dir)
    generator.generate_dataset(num_samples=args.samples)
