"""
================================================================================
DESCRIPTION:
    Generates synthetic chair images with simulated damages for training.
    Saves images and a simple JSON annotations file.

USAGE:
    python scripts/generate_synthetic_data.py --samples 200 --output_dir ./data/synthetic_damage/

OUTPUTS:
    ./data/synthetic_damage/images/*.jpg
    ./data/synthetic_damage/annotations.json

ARGUMENTS:
    --samples     Number of images to generate (default 100)
    --output_dir  Output directory (default ./data/synthetic_damage/)
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from tqdm import tqdm


class SyntheticDataGenerator:
    def __init__(self, output_dir: str = "./data/synthetic_damage/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        self.chair_parts = [
            "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
        ]
        self.damage_types = ["missing", "cracked", "broken", "loose", "scratched"]

        # Augmentations (Color/Noise only)
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4), # Increased noise
            A.Blur(blur_limit=3, p=0.2),
        ])

    def _canonical_parts(self, width: int, height: int) -> Dict[str, List[int]]:
        """Define canonical chair shape."""
        base_w, base_h = 640.0, 480.0
        base_parts = {
            "seat": [200, 200, 440, 250],
            "back": [220, 100, 420, 200],
            "front_left_leg": [200, 250, 230, 350],
            "front_right_leg": [410, 250, 440, 350],
            "back_left_leg": [220, 250, 250, 350],
            "back_right_leg": [390, 250, 420, 350],
            "armrest_left": [180, 150, 220, 180],
            "armrest_right": [420, 150, 460, 180],
        }

        sx = width / base_w
        sy = height / base_h
        scaled = {}
        for name, (x1, y1, x2, y2) in base_parts.items():
            scaled[name] = [
                int(round(x1 * sx)), int(round(y1 * sy)),
                int(round(x2 * sx)), int(round(y2 * sy)),
            ]
        return scaled

    def _apply_global_transform(self, parts, width, height):
        """Random scaling/shifting."""
        xs = [c for p in parts.values() for c in (p[0], p[2])]
        ys = [c for p in parts.values() for c in (p[1], p[3])]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))

        scale = random.uniform(0.80, 1.25) # More variation
        dx = random.randint(-50, 50)
        dy = random.randint(-40, 40)

        new_parts = {}
        for name, (x1, y1, x2, y2) in parts.items():
            x1p = (x1 - cx) * scale + cx + dx
            x2p = (x2 - cx) * scale + cx + dx
            y1p = (y1 - cy) * scale + cy + dy
            y2p = (y2 - cy) * scale + cy + dy

            # Clamp
            x1p, x2p = sorted([max(0, min(width-1, v)) for v in (x1p, x2p)])
            y1p, y2p = sorted([max(0, min(height-1, v)) for v in (y1p, y2p)])
            
            new_parts[name] = [int(x1p), int(y1p), int(x2p), int(y2p)]
        return new_parts

    def _base_canvas(self, width=640, height=480):
        # Draw background
        base_color = random.randint(200, 255)
        bg = np.full((height, width, 3), base_color, dtype=np.uint8)
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)

        canonical = self._canonical_parts(width, height)
        parts = self._apply_global_transform(canonical, width, height)

        # Draw parts (Distinct Colors to help model separate parts)
        colors = {
            "seat": "navajowhite", "back": "burlywood",
            "front_left_leg": "peru", "front_right_leg": "peru",
            "back_left_leg": "saddlebrown", "back_right_leg": "saddlebrown",
            "armrest_left": "tan", "armrest_right": "tan"
        }

        for name, bb in parts.items():
            draw.rectangle(bb, outline="black", width=3, fill=colors.get(name, "gray"))

        return img, parts

    def _damage_subbox(self, part_box):
        x1, y1, x2, y2 = part_box
        pw, ph = x2 - x1, y2 - y1
        if pw <= 10 or ph <= 10: return part_box

        # Damage is 40-80% of part size
        scale_w = random.uniform(0.4, 0.8)
        scale_h = random.uniform(0.4, 0.8)
        dw, dh = pw * scale_w, ph * scale_h

        sx = random.uniform(x1, x2 - dw)
        sy = random.uniform(y1, y2 - dh)
        return [int(sx), int(sy), int(sx + dw), int(sy + dh)]

    def _apply_damage(self, img, parts, damage_info):
        draw = ImageDraw.Draw(img)
        damage_boxes = {}

        for d in damage_info:
            part = d["part"]
            typ = d["type"]
            if part not in parts: continue

            sub_box = self._damage_subbox(parts[part])
            x1, y1, x2, y2 = sub_box
            damage_boxes[(part, typ)] = sub_box

            # --- CRITICAL FIX: THICKER LINES & BOLDER COLORS ---
            if typ == "missing":
                # 'Missing' should look like the background (white/grey) + dashed outline
                draw.rectangle([x1, y1, x2, y2], fill="white", outline="black", width=1)
                # Add 'X' to signify void
                draw.line([x1, y1, x2, y2], fill="black", width=1)
                draw.line([x2, y1, x1, y2], fill="black", width=1)

            elif typ == "broken":
                # Broken = Red Snap
                draw.rectangle([x1, y1, x2, y2], fill="darkred")
                draw.line([x1, y1, x2, y2], fill="yellow", width=4) # Thick

            elif typ == "cracked":
                # Cracked = Black ZigZags
                for _ in range(4):
                    p1 = (random.randint(x1, x2), random.randint(y1, y2))
                    p2 = (p1[0] + random.randint(-20, 20), p1[1] + random.randint(-20, 20))
                    # FIX: Width 4 ensures it survives resizing
                    draw.line([p1, p2], fill="black", width=4) 

            elif typ == "scratched":
                # Scratched = Multiple Thin Lines (but thicker than before)
                for _ in range(8):
                    p1 = (random.randint(x1, x2), random.randint(y1, y2))
                    p2 = (p1[0] + random.randint(-30, 30), p1[1] + random.randint(-30, 30))
                    draw.line([p1, p2], fill="dimgray", width=2) # Was width=1

            elif typ == "loose":
                # Loose = Wobbly outline / Offset
                draw.rectangle([x1, y1, x2, y2], outline="orange", width=5) # Thick border

        return img, damage_boxes

    def generate_dataset(self, num_samples: int = 1000):
        ann_path = os.path.join(self.output_dir, "annotations.json")
        images_dir = os.path.join(self.output_dir, "images")
        annotations = []

        for i in tqdm(range(num_samples), desc="Generating Data"):
            img, parts = self._base_canvas()
            
            # 1-3 damages per image
            n_damage = random.randint(1, 3)
            damage_info = []
            
            # Ensure we select valid parts
            available_parts = list(parts.keys())
            if len(available_parts) >= n_damage:
                selected_parts = random.sample(available_parts, n_damage)
            else:
                selected_parts = available_parts

            for part in selected_parts:
                damage_info.append({
                    "part": part,
                    "type": random.choice(self.damage_types),
                    "severity": 1.0,
                })

            img, damage_boxes = self._apply_damage(img, parts, damage_info)

            # Convert to numpy for Albumentations
            img_np = np.array(img)
            img_np = self.augmentation(image=img_np)["image"]
            img_aug = Image.fromarray(img_np)

            fname = f"synthetic_{i:05d}.jpg"
            img_aug.save(os.path.join(images_dir, fname))

            damages_with_boxes = []
            for d in damage_info:
                key = (d["part"], d["type"])
                # Fallback to part box if subbox generation failed
                bbox = damage_boxes.get(key, parts.get(d["part"], [0,0,10,10]))
                damages_with_boxes.append({
                    "part": d["part"],
                    "type": d["type"],
                    "bbox": bbox
                })

            annotations.append({
                "filename": fname,
                "parts": parts,
                "damages": damages_with_boxes
            })

        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Generated {num_samples} samples.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=2000) # Increased default
    p.add_argument("--output_dir", type=str, default="./data/synthetic_damage/")
    args = p.parse_args()

    SyntheticDataGenerator(args.output_dir).generate_dataset(args.samples)