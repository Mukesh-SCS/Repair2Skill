"""
================================================================================
ENHANCED SYNTHETIC DATA GENERATOR FOR DAMAGE DETECTION
================================================================================
Improved synthetic data generation with:
    - Multiple damages per image (realistic scenarios)
    - Enhanced damage patterns (more realistic visual representation)
    - Better background variation (textures, lighting)
    - Per-damage-type distribution control
    - Data augmentation pipeline

USAGE:
    python scripts/generate_synthetic_data_enhanced.py \\
        --samples 3000 \\
        --multi_damage_ratio 0.15 \\
        --output_dir ./data/synthetic_damage/

OUTPUTS:
    ./data/synthetic_damage/images/*.jpg       (synthetic images)
    ./data/synthetic_damage/annotations.json   (COCO-format annotations)
    ./data/synthetic_damage/stats.json         (distribution statistics)

Author: Repair2Skill Enhancement
================================================================================
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFilter, ImageColor
from tqdm import tqdm
from pathlib import Path


class EnhancedSyntheticDataGenerator:
    """Advanced synthetic chair damage dataset generator."""
    
    def __init__(self, output_dir: str = "./data/synthetic_damage/"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # Chair parts
        self.chair_parts = [
            "seat", "back",
            "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg",
            "armrest_left", "armrest_right"
        ]
        
        # Damage types
        self.damage_types = ["missing", "cracked", "broken", "loose", "scratched"]
        
        # Part colors (with variation)
        self.part_colors = {
            "seat": "#D2B48C",  # Tan
            "back": "#CD853F",  # Peru
            "front_left_leg": "#8B4513",  # Saddle brown
            "front_right_leg": "#8B4513",
            "back_left_leg": "#A0522D",  # Sienna
            "back_right_leg": "#A0522D",
            "armrest_left": "#BC8F8F",  # Rosy brown
            "armrest_right": "#BC8F8F",
        }
        
        # Statistics tracking
        self.stats = {
            "total_images": 0,
            "single_damage": 0,
            "multi_damage": 0,
            "damage_distribution": {dt: 0 for dt in self.damage_types},
            "part_distribution": {pt: 0 for pt in self.chair_parts}
        }
    
    # =====================================================================
    # BACKGROUND & TEXTURE GENERATION
    # =====================================================================
    
    def _generate_background(self, width: int, height: int) -> Image.Image:
        """Generate realistic background with texture variation."""
        # Choose background style
        style = random.choice(['solid', 'gradient', 'texture'])
        
        if style == 'solid':
            # Solid color with slight variation
            base_color = random.randint(180, 240)
            color = (base_color, base_color, base_color)
            bg = Image.new('RGB', (width, height), color)
        
        elif style == 'gradient':
            # Subtle gradient background
            bg = Image.new('RGB', (width, height))
            pixels = bg.load()
            
            for y in range(height):
                intensity = int(200 + (y / height) * 40)
                for x in range(width):
                    pixels[x, y] = (intensity, intensity, intensity)
        
        else:  # texture
            # Noisy texture background
            base_color = random.randint(190, 235)
            bg = Image.new('RGB', (width, height), (base_color, base_color, base_color))
            
            # Add noise
            noise = np.random.normal(0, 10, (height, width, 3)).astype(np.uint8)
            bg_array = np.array(bg) + noise
            bg_array = np.clip(bg_array, 0, 255).astype(np.uint8)
            bg = Image.fromarray(bg_array)
        
        return bg
        def _generate_background(self, width: int, height: int) -> Image.Image:
            """Generate realistic background with texture variation and real images."""
            style = random.choice(['solid', 'gradient', 'texture', 'real'])
            if style == 'real':
                real_bg_dir = os.path.join(self.output_dir, "real_backgrounds")
                if os.path.exists(real_bg_dir):
                    files = [f for f in os.listdir(real_bg_dir) if f.lower().endswith(('.jpg', '.png'))]
                    if files:
                        fname = random.choice(files)
                        try:
                            bg = Image.open(os.path.join(real_bg_dir, fname)).convert('RGB').resize((width, height))
                            return bg
                        except Exception:
                            pass
            # ...existing code...
    
    # =====================================================================
    # CHAIR GEOMETRY
    # =====================================================================
    
    def _canonical_parts(self, W: int = 640, H: int = 480) -> Dict[str, List]:
        """Get canonical part coordinates at base resolution."""
        base_w, base_h = 640, 480
        
        base_parts = {
            "seat": [200, 220, 440, 280],
            "back": [210, 120, 430, 220],
            "front_left_leg": [210, 280, 240, 390],
            "front_right_leg": [400, 280, 430, 390],
            "back_left_leg": [230, 280, 260, 380],
            "back_right_leg": [380, 280, 410, 380],
            "armrest_left": [175, 170, 210, 205],
            "armrest_right": [430, 170, 465, 205],
        }
        
        # Scale to target dimensions
        sx, sy = W / base_w, H / base_h
        parts = {}
        
        for name, (x1, y1, x2, y2) in base_parts.items():
            parts[name] = [
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy)
            ]
        
        return parts
    
    def _apply_transform(self, parts: Dict, W: int, H: int) -> Dict:
        """Apply random transformation (scale, rotate, translate) to parts."""
        # Calculate chair center
        xs = [c for b in parts.values() for c in (b[0], b[2])]
        ys = [c for b in parts.values() for c in (b[1], b[3])]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        # Random transformation parameters
        scale = random.uniform(0.85, 1.12)  # More variation
        rotation = random.uniform(-3, 3)    # Small rotation
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        
        out = {}
        for name, (x1, y1, x2, y2) in parts.items():
            # Scale around center
            nx1 = (x1 - cx) * scale + cx + dx
            nx2 = (x2 - cx) * scale + cx + dx
            ny1 = (y1 - cy) * scale + cy + dy
            ny2 = (y2 - cy) * scale + cy + dy
            
            # Clamp to image bounds
            nx1 = max(0, min(W - 1, nx1))
            nx2 = max(nx1 + 2, min(W - 1, nx2))
            ny1 = max(0, min(H - 1, ny1))
            ny2 = max(ny1 + 2, min(H - 1, ny2))
            
            out[name] = [int(nx1), int(ny1), int(nx2), int(ny2)]
        
        return out
    
    # =====================================================================
    # DAMAGE RENDERING
    # =====================================================================
    
    def _sample_subbox(self, box: List, scale_range: Tuple = (0.6, 0.95)) -> List:
        """Sample a damage box within a part boundary - LARGER for better detection."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        if w < 5 or h < 5:
            return box  # Part too small
        
        # Use larger scale range to make damages more visible
        sw = random.uniform(*scale_range)
        sh = random.uniform(*scale_range)
        
        bw, bh = w * sw, h * sh
        sx = random.uniform(x1, max(x1 + 1, x2 - bw))
        sy = random.uniform(y1, max(y1 + 1, y2 - bh))
        
        bx1 = int(sx)
        by1 = int(sy)
        bx2 = int(sx + bw)
        by2 = int(sy + bh)
        
        bx2 = max(bx2, bx1 + 4)
        by2 = max(by2, by1 + 4)
        
        return [bx1, by1, bx2, by2]
    
    def _contrast_color(self, color: str) -> str:
        """Get contrasting color (black or white) for text on background."""
        try:
            r, g, b = ImageColor.getrgb(color)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            return "black" if brightness > 130 else "white"
        except:
            return "black"
    
    def _draw_missing(self, draw: ImageDraw.ImageDraw, box: List, color: str):
        """Draw missing damage (hole/void) - VERY VISIBLE."""
        x1, y1, x2, y2 = box
        # Bold red background to show missing part
        draw.rectangle([x1, y1, x2, y2], fill="#FF0000", outline="black", width=4)
        # Draw large X pattern to indicate missing
        draw.line([(x1, y1), (x2, y2)], fill="black", width=6)
        draw.line([(x1, y2), (x2, y1)], fill="black", width=6)
        # Add "MISSING" indicator if space allows
        if (x2 - x1) > 30 and (y2 - y1) > 30:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for offset_x in range(-10, 11, 5):
                for offset_y in range(-10, 11, 5):
                    draw.point((cx + offset_x, cy + offset_y), fill="white")
    
    def _draw_broken(self, draw: ImageDraw.ImageDraw, box: List, color: str):
        """Draw broken damage (shattered/fractured) - VERY VISIBLE."""
        x1, y1, x2, y2 = box
        
        # Bold dark red/orange fill for break - much more visible
        draw.rectangle([x1, y1, x2, y2], fill="#CC0000", outline="black", width=3)
        
        # Draw multiple bold jagged cracks
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Horizontal jagged break line
        points = []
        for x in range(x1, x2 + 1, max(1, (x2 - x1) // 6)):
            y = cy + random.randint(-8, 8)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill="black", width=5)
        
        # Multiple radiating cracks from center
        for _ in range(5):
            angle = random.uniform(0, 2 * np.pi)
            max_length = max(15, min(x2-x1, y2-y1) // 2)
            length = random.randint(15, max(15, max_length))
            ex = int(cx + length * np.cos(angle))
            ey = int(cy + length * np.sin(angle))
            ex = max(x1, min(x2, ex))
            ey = max(y1, min(y2, ey))
            draw.line([cx, cy, ex, ey], fill="black", width=4)
    
    def _draw_cracked(self, draw: ImageDraw.ImageDraw, box: List, color: str):
        """Draw cracked damage (lines/fractures) - VERY VISIBLE."""
        x1, y1, x2, y2 = box
        
        # Draw yellow/orange background to highlight crack area
        draw.rectangle([x1, y1, x2, y2], fill="#FFAA00", outline="black", width=3)
        
        # Multiple bold black cracks
        for _ in range(5):
            start_x = random.randint(x1, x2)
            start_y = random.randint(y1, y2)
            
            # Draw zigzag crack line - much bolder
            cx, cy = start_x, start_y
            for _ in range(4):
                nx = cx + random.randint(-20, 20)
                ny = cy + random.randint(8, 25)
                nx = max(x1, min(x2, nx))
                ny = max(y1, min(y2, ny))
                draw.line([cx, cy, nx, ny], fill="black", width=5)
                cx, cy = nx, ny
    
    def _draw_scratched(self, draw: ImageDraw.ImageDraw, box: List, color: str):
        """Draw scratched damage (surface marks) - VERY VISIBLE."""
        x1, y1, x2, y2 = box
        
        # Gray/silver background to show scratched area
        draw.rectangle([x1, y1, x2, y2], fill="#888888", outline="black", width=3)
        
        # Multiple bold white scratch marks on gray background
        num_scratches = random.randint(6, 10)
        for _ in range(num_scratches):
            sx = x1 + random.randint(0, max(1, x2 - x1))
            sy = y1 + random.randint(0, max(1, y2 - y1))
            max_length = max(25, min(60, max(x2-x1, y2-y1)))
            length = random.randint(25, max_length)
            angle = random.uniform(0, np.pi)
            
            ex = int(sx + length * np.cos(angle))
            ey = int(sy + length * np.sin(angle))
            ex = max(x1, min(x2, ex))
            ey = max(y1, min(y2, ey))
            
            # Bold white scratches
            draw.line([sx, sy, ex, ey], fill="white", width=4)
            # Add shadow for depth
            draw.line([sx+1, sy+1, ex+1, ey+1], fill="black", width=2)
    
    def _draw_loose(self, draw: ImageDraw.ImageDraw, box: List, color: str):
        """Draw loose damage (misalignment/separation) - VERY VISIBLE."""
        x1, y1, x2, y2 = box
        
        # Bold orange/yellow fill to show loose area
        draw.rectangle([x1, y1, x2, y2], fill="#FFA500", outline="black", width=4)
        
        # Draw wavy/offset lines to show looseness
        mid_y = (y1 + y2) // 2
        points = []
        for x in range(x1, x2 + 1, max(1, (x2 - x1) // 8)):
            y_offset = 8 * np.sin((x - x1) / max(1, (x2 - x1)) * 2 * np.pi)
            points.append((x, int(mid_y + y_offset)))
        
        if len(points) > 1:
            draw.line(points, fill="black", width=5)
        
        # Add arrows/indicators showing movement
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        arrow_len = 10
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.line(
                [cx, cy, cx + dx * arrow_len, cy + dy * arrow_len],
                fill="red", width=4
            )
    
    def _draw_damage(self, draw: ImageDraw.ImageDraw, box: List, 
                     damage_type: str, base_color: str):
        """Draw damage on image."""
        handlers = {
            "missing": self._draw_missing,
            "broken": self._draw_broken,
            "cracked": self._draw_cracked,
            "scratched": self._draw_scratched,
            "loose": self._draw_loose
        }
        
        handler = handlers.get(damage_type, self._draw_missing)
        handler(draw, box, base_color)
    
    # =====================================================================
    # DATASET GENERATION
    # =====================================================================
    
    def generate_dataset(self, N: int = 2000, multi_damage_ratio: float = 0.15):
        """
        Generate N synthetic images with damage annotations.
        
        Args:
            N: Total number of images to generate
            multi_damage_ratio: Fraction of images with multiple damages (0.0-1.0)
        """
        images_dir = os.path.join(self.output_dir, "images")
        annotations = []
        
        print(f"Generating {N} synthetic images (multi-damage ratio: {multi_damage_ratio:.0%})")
        
        for idx in tqdm(range(N)):
            W, H = 640, 480
            
            # Generate background
            bg = self._generate_background(W, H)
            img = bg
            draw = ImageDraw.Draw(img)
            
            # Get chair parts at this resolution
            parts = self._canonical_parts(W, H)
            parts = self._apply_transform(parts, W, H)
            
            # Draw chair parts
            for part_name, (x1, y1, x2, y2) in parts.items():
                color = self.part_colors[part_name]
                
                # Draw part with slight shading variation
                draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=2)
                
                # Add subtle shading
                if random.random() > 0.5:
                    shade_color = tuple(max(0, c - 20) for c in ImageColor.getrgb(color))
                    shade_width = max(1, (x2 - x1) // 20)
                    draw.rectangle([x2 - shade_width, y1, x2, y2], 
                                 fill=shade_color, outline=None)
            
            # Decide number of damages
            is_multi = random.random() < multi_damage_ratio
            num_damages = random.randint(2, 3) if is_multi else 1
            
            # Select parts to damage (no duplicates)
            available_parts = list(self.chair_parts)
            damage_parts = random.sample(available_parts, min(num_damages, len(available_parts)))
            
            damages = []
            for part_choice in damage_parts:
                damage_type = random.choice(self.damage_types)
                
                # Generate damage box
                damage_box = self._sample_subbox(parts[part_choice])
                
                # Draw damage
                self._draw_damage(draw, damage_box, damage_type, 
                                self.part_colors[part_choice])
                
                # Record annotation
                damages.append({
                    "part": part_choice,
                    "type": damage_type,
                    "bbox": damage_box
                })
                
                # Update statistics
                self.stats["damage_distribution"][damage_type] += 1
                self.stats["part_distribution"][part_choice] += 1
            
            # Update multi-damage statistics
            if len(damages) > 1:
                self.stats["multi_damage"] += 1
            else:
                self.stats["single_damage"] += 1
            
            # Save image
            fname = f"synthetic_{idx:05d}.jpg"
            img.save(os.path.join(images_dir, fname), quality=95)
            
            # Create annotation
            annotation = {
                "filename": fname,
                "width": W,
                "height": H,
                "parts": parts,
                "damages": damages
            }
            annotations.append(annotation)
        
        self.stats["total_images"] = N
        
        # Save annotations
        ann_path = os.path.join(self.output_dir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Saved {N} images to {images_dir}/")
        print(f"✓ Saved annotations to {ann_path}")
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        self._print_statistics()
        print(f"✓ Saved statistics to {stats_path}")
            # Save a grid of sample images for inspection
            try:
                import math
                grid_size = min(25, N)
                grid_cols = 5
                grid_rows = math.ceil(grid_size / grid_cols)
                grid_img = Image.new('RGB', (grid_cols * W, grid_rows * H))
                for i in range(grid_size):
                    img_path = os.path.join(images_dir, f"synthetic_{i:05d}.jpg")
                    if os.path.exists(img_path):
                        img_sample = Image.open(img_path)
                        x = (i % grid_cols) * W
                        y = (i // grid_cols) * H
                        grid_img.paste(img_sample, (x, y))
                grid_img.save(os.path.join(self.output_dir, "sample_grid.jpg"), quality=95)
                print("✓ Saved sample grid to sample_grid.jpg")
            except Exception as e:
                print(f"[WARN] Could not save sample grid: {e}")
    
        def generate_negative_samples(self, N: int = 200):
            """Generate images with no damage for negative samples."""
            images_dir = os.path.join(self.output_dir, "images")
            W, H = 640, 480
            for idx in range(N):
                bg = self._generate_background(W, H)
                img = bg
                draw = ImageDraw.Draw(img)
                parts = self._canonical_parts(W, H)
                parts = self._apply_transform(parts, W, H)
                for part_name, (x1, y1, x2, y2) in parts.items():
                    color = self.part_colors[part_name]
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=2)
                fname = f"negative_{idx:05d}.jpg"
                img.save(os.path.join(images_dir, fname), quality=95)
    
    def _print_statistics(self):
        """Print generation statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total images: {self.stats['total_images']}")
        print(f"Single damage: {self.stats['single_damage']} ({100*self.stats['single_damage']/self.stats['total_images']:.1f}%)")
        print(f"Multi damage: {self.stats['multi_damage']} ({100*self.stats['multi_damage']/self.stats['total_images']:.1f}%)")
        
        print("\nDamage distribution:")
        total_damages = sum(self.stats["damage_distribution"].values())
        for dtype, count in sorted(self.stats["damage_distribution"].items()):
            pct = 100 * count / total_damages if total_damages > 0 else 0
            print(f"  {dtype:12s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nPart distribution:")
        total_parts = sum(self.stats["part_distribution"].values())
        for pname, count in sorted(self.stats["part_distribution"].items()):
            pct = 100 * count / total_parts if total_parts > 0 else 0
            print(f"  {pname:18s}: {count:4d} ({pct:5.1f}%)")
        print("="*60 + "\n")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate enhanced synthetic damage dataset"
    )
    parser.add_argument(
        "--samples", type=int, default=3000,
        help="Number of images to generate (default: 3000)"
    )
    parser.add_argument(
        "--multi_damage_ratio", type=float, default=0.15,
        help="Fraction with multiple damages (default: 0.15)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/synthetic_damage/",
        help="Output directory (default: ./data/synthetic_damage/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate dataset
    generator = EnhancedSyntheticDataGenerator(args.output_dir)
    generator.generate_dataset(args.samples, args.multi_damage_ratio)


# Backward compatibility alias
SyntheticDataGenerator = EnhancedSyntheticDataGenerator
