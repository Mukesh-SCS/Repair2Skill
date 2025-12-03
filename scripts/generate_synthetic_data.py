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
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm


class SyntheticDataGenerator:
    def __init__(self, output_dir="./data/synthetic_damage/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

        self.chair_parts = [
            "seat", "back",
            "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg",
            "armrest_left", "armrest_right"
        ]

        # Keep all 5 damage types
        self.damage_types = ["missing", "cracked", "broken", "loose", "scratched"]

        # Colors
        self.part_colors = {
            "seat": "navajowhite",
            "back": "burlywood",
            "front_left_leg": "peru",
            "front_right_leg": "peru",
            "back_left_leg": "saddlebrown",
            "back_right_leg": "saddlebrown",
            "armrest_left": "tan",
            "armrest_right": "tan",
        }

    # -------------------------------------------------------------
    # Chair geometry
    # -------------------------------------------------------------
    def _canonical_parts(self, W=640, H=480):
        base_w, base_h = 640, 480

        base = {
            "seat": [200, 220, 440, 280],
            "back": [210, 120, 430, 220],
            "front_left_leg": [210, 280, 240, 390],
            "front_right_leg": [400, 280, 430, 390],
            "back_left_leg": [230, 280, 260, 380],
            "back_right_leg": [380, 280, 410, 380],
            "armrest_left": [175, 170, 210, 205],
            "armrest_right": [430, 170, 465, 205],
        }

        sx, sy = W / base_w, H / base_h
        parts = {}

        for name, (x1, y1, x2, y2) in base.items():
            parts[name] = [
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy)
            ]

        return parts

    def _apply_transform(self, parts, W, H):
        xs = [c for b in parts.values() for c in (b[0], b[2])]
        ys = [c for b in parts.values() for c in (b[1], b[3])]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        scale = random.uniform(0.90, 1.08)
        dx, dy = random.randint(-20, 20), random.randint(-20, 20)

        out = {}
        for name, (x1, y1, x2, y2) in parts.items():
            nx1 = (x1 - cx) * scale + cx + dx
            nx2 = (x2 - cx) * scale + cx + dx
            ny1 = (y1 - cy) * scale + cy + dy
            ny2 = (y2 - cy) * scale + cy + dy

            nx1, nx2 = sorted([max(0, min(W-1, nx1)), max(0, min(W-1, nx2))])
            ny1, ny2 = sorted([max(0, min(H-1, ny1)), max(0, min(H-1, ny2))])

            out[name] = [int(nx1), int(ny1), int(nx2), int(ny2)]
        return out

    # -------------------------------------------------------------
    # Damage helpers
    # -------------------------------------------------------------
    def _sample_subbox(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        # 30–60% of part size
        sw = random.uniform(0.50, 0.80)
        sh = random.uniform(0.50, 0.80)

        bw, bh = w * sw, h * sh
        sx = random.uniform(x1, x2 - bw)
        sy = random.uniform(y1, y2 - bh)

        bx1 = int(sx)
        by1 = int(sy)
        bx2 = int(sx + bw)
        by2 = int(sy + bh)

        # safety
        bx2 = max(bx2, bx1 + 4)
        by2 = max(by2, by1 + 4)
        return [bx1, by1, bx2, by2]

    def _contrast(self, color):
        try:
            r, g, b = ImageColor.getrgb(color)
        except:
            r = g = b = 150
        return "black" if (r + g + b) / 3 > 130 else "white"

    def _draw_damage(self, draw, box, dmg, base_color):
        x1, y1, x2, y2 = box
        contrast = self._contrast(base_color)

        # --- simple, SSD-friendly patterns ---
        if dmg == "missing":
            draw.rectangle([x1, y1, x2, y2], fill="white", outline="black", width=6)

        elif dmg == "broken":
            draw.rectangle([x1, y1, x2, y1 + int((y2 - y1) * 0.5)], fill="darkred")
            for x in range(x1, x2, 10):
                draw.line([x, y1 + 20, x + 8, y1 + 20 + random.randint(-6, 6)],
                          fill="yellow", width=6)

        elif dmg == "cracked":
            cx, cy = (x1 + x2) // 2, y1
            for _ in range(5):
                nx = cx + random.randint(-20, 20)
                ny = cy + random.randint(15, 30)
                draw.line([cx, cy, nx, ny], fill=contrast, width=10)
                cx, cy = nx, ny

        elif dmg == "scratched":
            for k in range(3):
                oy = random.randint(-6, 6)
                draw.line([x1, y1 + oy + k * 15, x2, y1 + oy + k * 15],
                          fill="dimgray", width=10)

        elif dmg == "loose":
            draw.rectangle([x1, y1, x2, y2], outline="orange", width=10)
            # inner safe border
            if x2 - x1 > 20 and y2 - y1 > 20:
                draw.rectangle([x1+8, y1+8, x2-8, y2-8], outline="orange", width=6)

    # -------------------------------------------------------------
    # MAIN
    # -------------------------------------------------------------
    def generate_dataset(self, N=1000):
        images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        ann = []

        for i in tqdm(range(N), desc="Generating Synthetic Data"):
            W, H = 640, 480

            # Create blank background + chair parts
            bg_color = random.randint(210, 240)
            bg = np.full((H, W, 3), bg_color, dtype=np.uint8)
            img = Image.fromarray(bg)
            draw = ImageDraw.Draw(img)

            parts = self._canonical_parts(W, H)
            parts = self._apply_transform(parts, W, H)

            # Draw chair
            for name, bb in parts.items():
                draw.rectangle(bb, fill=self.part_colors[name], outline="black", width=4)

            # --------------------------------------------------
            # EXACTLY ONE DAMAGE PER IMAGE
            # --------------------------------------------------
            part_choice = random.choice(self.chair_parts)
            damage_type = random.choice(self.damage_types)

            # Generate local box
            damage_box = self._sample_subbox(parts[part_choice])

            # Draw the damage
            self._draw_damage(draw, damage_box, damage_type, self.part_colors[part_choice])

            # Save image + annotation
            fname = f"synthetic_{i:05d}.jpg"
            img.save(os.path.join(images_dir, fname))

            ann.append({
                "filename": fname,
                "parts": parts,
                "damages": [{
                    "part": part_choice,
                    "type": damage_type,
                    "bbox": damage_box
                }]
            })

        with open(os.path.join(self.output_dir, "annotations.json"), "w") as f:
            json.dump(ann, f, indent=2)

        print(f"[OK] Generated {N} synthetic images.")
        

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--output_dir", type=str, default="./data/synthetic_damage/")
    args = p.parse_args()

    SyntheticDataGenerator(args.output_dir).generate_dataset(args.samples)
