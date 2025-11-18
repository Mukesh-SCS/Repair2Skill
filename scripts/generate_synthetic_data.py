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

        # Only photometric augmentations (no geometry) so boxes stay valid
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ])

    # ------------------------------------------------------------------
    # Canonical chair layout in a normalized coordinate frame
    # ------------------------------------------------------------------
    def _canonical_parts(self, width: int, height: int) -> Dict[str, List[int]]:
        """
        Define a simple canonical chair shape (seat, back, legs, armrests)
        in a central region of the image.
        """
        # We'll design coordinates in a nominal 640x480 frame, then
        # rescale to the requested width/height.
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
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
            ]
        return scaled

    def _apply_global_transform(
        self,
        parts: Dict[str, List[int]],
        width: int,
        height: int
    ) -> Dict[str, List[int]]:
        """
        Apply a mild global scale + translation to the whole chair
        so that we get some variation but preserve structure.
        """
        # Compute bounding box of the whole canonical chair
        xs = []
        ys = []
        for (x1, y1, x2, y2) in parts.values():
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)

        # Mild scale + random translation
        scale = random.uniform(0.85, 1.15)
        dx = random.randint(-30, 30)
        dy = random.randint(-20, 20)

        new_parts = {}
        for name, (x1, y1, x2, y2) in parts.items():
            # scale around center, then translate
            x1p = (x1 - cx) * scale + cx + dx
            x2p = (x2 - cx) * scale + cx + dx
            y1p = (y1 - cy) * scale + cy + dy
            y2p = (y2 - cy) * scale + cy + dy

            # clamp and sort
            x1p, x2p = sorted([
                max(0, min(width - 1, x1p)),
                max(0, min(width - 1, x2p)),
            ])
            y1p, y2p = sorted([
                max(0, min(height - 1, y1p)),
                max(0, min(height - 1, y2p)),
            ])

            new_parts[name] = [
                int(round(x1p)),
                int(round(y1p)),
                int(round(x2p)),
                int(round(y2p)),
            ]

        return new_parts

    def _base_canvas(
        self,
        width: int = 640,
        height: int = 480
    ) -> Tuple[Image.Image, Dict[str, List[int]]]:
        """
        Create a plain background and draw a structured chair with
        slightly randomized global transform.
        """
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)

        canonical = self._canonical_parts(width, height)
        parts = self._apply_global_transform(canonical, width, height)

        # Draw each part as a filled lightgray rectangle + outline
        for _, bb in parts.items():
            draw.rectangle(bb, outline="black", width=2, fill="lightgray")

        return img, parts

    def _apply_damage(self, img, parts, damage_info):
        draw = ImageDraw.Draw(img)

        for d in damage_info:
            part = d["part"]
            typ = d["type"]
            if part not in parts:
                continue

            x1, y1, x2, y2 = parts[part]

            if typ == "missing":
                draw.rectangle([x1, y1, x2, y2], fill="white", outline="white")

            elif typ == "broken":
                draw.rectangle([x1, y1, x2, y2], fill="red")
                for _ in range(5):
                    p1 = (random.randint(x1, x2), random.randint(y1, y2))
                    p2 = (
                        p1[0] + random.randint(-30, 30),
                        p1[1] + random.randint(-30, 30),
                    )
                    draw.line([p1, p2], fill="yellow", width=3)

            elif typ == "cracked":
                for _ in range(6):
                    p1 = (random.randint(x1, x2), random.randint(y1, y2))
                    p2 = (
                        p1[0] + random.randint(-40, 40),
                        p1[1] + random.randint(-40, 40),
                    )
                    draw.line([p1, p2], fill="red", width=3)

            elif typ == "scratched":
                for _ in range(10):
                    p1 = (random.randint(x1, x2), random.randint(y1, y2))
                    p2 = (random.randint(x1, x2), random.randint(y1, y2))
                    draw.line([p1, p2], fill="brown", width=2)

            elif typ == "loose":
                draw.rectangle([x1, y1, x2, y2], outline="orange", width=6)

        return img

    def generate_dataset(self, num_samples: int = 1000):
        ann_path = os.path.join(self.output_dir, "annotations.json")
        images_dir = os.path.join(self.output_dir, "images")
        annotations = []

        for i in tqdm(range(num_samples), desc="Generating synthetic data"):
            img, parts = self._base_canvas()
            n_damage = random.randint(1, 3)

            damage_info = []
            for part in random.sample(self.chair_parts, n_damage):
                damage_info.append({
                    "part": part,
                    "type": random.choice(self.damage_types),
                    "severity": round(random.uniform(0.3, 1.0), 2),
                })

            img = self._apply_damage(img, parts, damage_info)

            # Photometric augmentations only (no geometry)
            img_np = np.array(img)
            img_np = self.augmentation(image=img_np)["image"]
            img_aug = Image.fromarray(img_np)

            fname = f"synthetic_{i:05d}.jpg"
            img_aug.save(os.path.join(images_dir, fname))

            annotations.append({
                "image_id": i,
                "filename": fname,
                "width": img_aug.width,
                "height": img_aug.height,
                "damages": damage_info,
                "parts": parts,
            })

        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)

        print(f"Generated {num_samples} images and {ann_path}")



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate synthetic chair damage dataset")
    p.add_argument("--samples", type=int, default=100, help="Number of samples")
    p.add_argument("--output_dir", type=str, default="./data/synthetic_damage/", help="Output directory")
    args = p.parse_args()

    gen = SyntheticDataGenerator(output_dir=args.output_dir)
    gen.generate_dataset(num_samples=args.samples)