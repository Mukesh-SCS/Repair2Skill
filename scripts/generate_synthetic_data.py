# scripts/generate_synthetic_data.py
# V2: Part boxes + part_damage. Domain randomization: rotation, perspective, shading,
#     edge-crossing damage, shadows, occlusion. Bboxes are updated to match transforms.
import math
import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageColor
from tqdm import tqdm

PARTS = [
    "seat", "back",
    "front_left_leg", "front_right_leg",
    "back_left_leg", "back_right_leg",
    "armrest_left", "armrest_right",
]

DAMAGE_TYPES = ["none", "missing", "cracked", "broken", "loose", "scratched"]


class SyntheticChairGenV2:
    def __init__(self, output_dir="./data/synth_v2"):
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.part_colors = {
            "seat": "#C8AA7A",
            "back": "#B07A3A",
            "front_left_leg": "#6C3A16",
            "front_right_leg": "#6C3A16",
            "back_left_leg": "#7A3F18",
            "back_right_leg": "#7A3F18",
            "armrest_left": "#A88A8A",
            "armrest_right": "#A88A8A",
        }

    def _bg(self, W, H):
        mode = random.choice(["solid", "gradient", "noise"])
        if mode == "solid":
            c = random.randint(170, 240)
            return Image.new("RGB", (W, H), (c, c, c))
        if mode == "gradient":
            img = Image.new("RGB", (W, H))
            px = img.load()
            a = random.randint(170, 210)
            b = random.randint(210, 245)
            for y in range(H):
                t = y / max(1, H - 1)
                v = int(a * (1 - t) + b * t)
                for x in range(W):
                    px[x, y] = (v, v, v)
            return img
        base = np.full((H, W, 3), random.randint(175, 235), dtype=np.uint8)
        noise = np.random.normal(0, 10, (H, W, 3)).astype(np.int16)
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _canonical_parts(self, W=640, H=480):
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
        sx, sy = W / 640, H / 480
        return {k: [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)] for k, (x1, y1, x2, y2) in base.items()}

    def _transform(self, parts, W, H):
        xs = [c for b in parts.values() for c in (b[0], b[2])]
        ys = [c for b in parts.values() for c in (b[1], b[3])]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        scale = random.uniform(0.85, 1.15)
        dx = random.randint(-25, 25)
        dy = random.randint(-25, 25)
        out = {}
        for name, (x1, y1, x2, y2) in parts.items():
            nx1 = (x1 - cx) * scale + cx + dx
            nx2 = (x2 - cx) * scale + cx + dx
            ny1 = (y1 - cy) * scale + cy + dy
            ny2 = (y2 - cy) * scale + cy + dy
            nx1 = max(0, min(W - 2, nx1))
            nx2 = max(nx1 + 2, min(W - 1, nx2))
            ny1 = max(0, min(H - 2, ny1))
            ny2 = max(ny1 + 2, min(H - 1, ny2))
            out[name] = [int(nx1), int(ny1), int(nx2), int(ny2)]
        return out

    def _rotate_boxes(self, parts, W, H, angle_deg):
        """Transform part boxes by the same rotation as img.rotate(-angle_deg).
        Image was rotated by -angle_deg around (W/2, H/2); so point (px,py) moved to R_{-angle}(px,py)."""
        cx, cy = W / 2, H / 2
        rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = {}
        for name, (x1, y1, x2, y2) in parts.items():
            xs, ys = [], []
            for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                dx, dy = px - cx, py - cy
                nx = cx + dx * cos_a - dy * sin_a
                ny = cy + dx * sin_a + dy * cos_a
                xs.append(nx)
                ys.append(ny)
            nx1 = max(0, min(W - 1, int(math.floor(min(xs)))))
            ny1 = max(0, min(H - 1, int(math.floor(min(ys)))))
            nx2 = max(nx1 + 1, min(W, int(math.ceil(max(xs)))))
            ny2 = max(ny1 + 1, min(H, int(math.ceil(max(ys)))))
            out[name] = [nx1, ny1, nx2, ny2]
        return out

    def _homography_from_quad_to_rect(self, tl, bl, br, tr, W, H):
        """Compute 3x3 homography H that maps quad (tl,bl,br,tr) -> (0,0), (0,H), (W,H), (W,0)."""
        src = np.array([tl, bl, br, tr], dtype=np.float64)
        dst = np.array([[0, 0], [0, H], [W, H], [W, 0]], dtype=np.float64)
        A = []
        for i in range(4):
            x, y = src[i, 0], src[i, 1]
            u, v = dst[i, 0], dst[i, 1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]
        return H

    def _apply_perspective_boxes(self, parts, H_mat, W, H_img):
        """Transform each part box's 4 corners by homography H; return new axis-aligned boxes."""
        out = {}
        for name, (x1, y1, x2, y2) in parts.items():
            xs, ys = [], []
            for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                p = np.array([px, py, 1.0])
                q = H_mat @ p
                q = q / q[2]
                xs.append(q[0])
                ys.append(q[1])
            nx1 = max(0, min(W - 1, int(math.floor(min(xs)))))
            ny1 = max(0, min(H_img - 1, int(math.floor(min(ys)))))
            nx2 = max(nx1 + 1, min(W, int(math.ceil(max(xs)))))
            ny2 = max(ny1 + 1, min(H_img, int(math.ceil(max(ys)))))
            out[name] = [nx1, ny1, nx2, ny2]
        return out

    def _apply_perspective(self, img, parts, W, H):
        """Slight trapezoid warp: source quad -> rectangle. Updates part boxes via homography."""
        max_shift = 14
        tl_x, tl_y = random.randint(0, max_shift), random.randint(0, max_shift)
        bl_x, bl_y = random.randint(0, max_shift), H - 1 - random.randint(0, max_shift)
        br_x, br_y = W - 1 - random.randint(0, max_shift), H - 1 - random.randint(0, max_shift)
        tr_x, tr_y = W - 1 - random.randint(0, max_shift), random.randint(0, max_shift)
        data = (tl_x, tl_y, bl_x, bl_y, br_x, br_y, tr_x, tr_y)
        try:
            out = img.transform(img.size, Image.QUAD, data, resample=Image.BICUBIC)
        except Exception:
            return img, parts, W, H
        if out is None:
            return img, parts, W, H
        try:
            H_mat = self._homography_from_quad_to_rect(
            (tl_x, tl_y), (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y), W, H
            )
            new_parts = self._apply_perspective_boxes(parts, H_mat, W, H)
            return out, new_parts, W, H
        except Exception:
            return img, parts, W, H

    def _apply_depth_shading(self, img):
        """Add simple depth-like shading (darker on one side)."""
        W, H = img.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Darken one corner/side
        which = random.choice(["top", "bottom", "left", "right"])
        alpha = random.randint(15, 40)
        if which == "top":
            for y in range(H):
                a = int(alpha * (1 - y / H))
                draw.line([(0, y), (W, y)], fill=(0, 0, 0, a))
        elif which == "bottom":
            for y in range(H):
                a = int(alpha * (y / H))
                draw.line([(0, y), (W, y)], fill=(0, 0, 0, a))
        elif which == "left":
            for x in range(W):
                a = int(alpha * (1 - x / W))
                draw.line([(x, 0), (x, H)], fill=(0, 0, 0, a))
        else:
            for x in range(W):
                a = int(alpha * (x / W))
                draw.line([(x, 0), (x, H)], fill=(0, 0, 0, a))
        return Image.alpha_composite(img, overlay)

    def _apply_shadow(self, img, W, H):
        """Draw a soft shadow (ellipse or blob) somewhere."""
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        xc = random.randint(W // 4, 3 * W // 4)
        yc = random.randint(H // 4, 3 * H // 4)
        rw = random.randint(30, 80)
        rh = random.randint(20, 50)
        draw.ellipse([xc - rw, yc - rh, xc + rw, yc + rh], fill=(0, 0, 0, random.randint(25, 55)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(8, 20)))
        return Image.alpha_composite(img, overlay)

    def _apply_occlusion(self, img, parts, W, H):
        """Random partial occlusion: small rectangle with background-like color over the image."""
        num = random.randint(0, 2)
        for _ in range(num):
            x1 = random.randint(0, W - 40)
            y1 = random.randint(0, H - 40)
            w = random.randint(15, 50)
            h = random.randint(15, 50)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)
            c = random.randint(160, 230)
            color = (c, c, c, random.randint(200, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        return img

    def _subbox(self, box, smin=0.25, smax=0.7, extend=0, img_bounds=(640, 480)):
        """Subregion inside box. If extend>0, allow region to extend outside box (for edge-crossing damage)."""
        x1, y1, x2, y2 = box
        Wb, Hb = img_bounds
        w = max(4, x2 - x1)
        h = max(4, y2 - y1)
        sw = random.uniform(smin, smax)
        sh = random.uniform(smin, smax)
        bw = max(8, int(w * sw))
        bh = max(8, int(h * sh))
        sx = random.randint(int(x1), max(int(x1), int(x2) - bw))
        sy = random.randint(int(y1), max(int(y1), int(y2) - bh))
        if extend > 0:
            ex = random.randint(0, min(extend, max(1, bw // 2)))
            ey = random.randint(0, min(extend, max(1, bh // 2)))
            return [
                max(0, sx - ex), max(0, sy - ey),
                min(Wb, sx + bw + ex), min(Hb, sy + bh + ey),
            ]
        return [sx, sy, sx + bw, sy + bh]

    def _draw_cracks(self, img, box):
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box
        for _ in range(random.randint(2, 5)):
            pts = []
            px = random.randint(x1, x2)
            py = random.randint(y1, y2)
            pts.append((px, py))
            for _ in range(random.randint(3, 6)):
                px = max(x1, min(x2, px + random.randint(-20, 20)))
                py = max(y1, min(y2, py + random.randint(8, 25)))
                pts.append((px, py))
            draw.line(pts, fill=(20, 20, 20), width=random.randint(2, 4))

    def _draw_scratches(self, img, box):
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box
        for _ in range(random.randint(6, 12)):
            sx = random.randint(x1, x2)
            sy = random.randint(y1, y2)
            ex = max(x1, min(x2, sx + random.randint(-60, 60)))
            ey = max(y1, min(y2, sy + random.randint(-30, 30)))
            draw.line([sx, sy, ex, ey], fill=(230, 230, 230), width=random.randint(1, 2))

    def _draw_broken(self, img, box):
        x1, y1, x2, y2 = box
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        poly = [(random.randint(x1, x2), random.randint(y1, y2)) for _ in range(8)]
        d.polygon(poly, fill=(30, 30, 30, 160))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
        return Image.alpha_composite(img, overlay.convert("RGBA"))

    def _apply_missing(self, img, part_box):
        x1, y1, x2, y2 = part_box
        patch = self._subbox(part_box, 0.35, 0.9)
        px1, py1, px2, py2 = patch
        h, w = py2 - py1, px2 - px1
        if h < 1 or w < 1:
            return
        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
        base = np.full((h, w, 3), 200, dtype=np.int16)
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(arr), (px1, py1))

    def generate(self, N=5000, val_ratio=0.2, multi_damage_ratio=0.2, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        W, H = 640, 480
        anns = []
        for i in tqdm(range(N)):
            img = self._bg(W, H).convert("RGBA")
            draw = ImageDraw.Draw(img)
            parts = self._transform(self._canonical_parts(W, H), W, H)

            for p, (x1, y1, x2, y2) in parts.items():
                c = ImageColor.getrgb(self.part_colors[p])
                jitter = random.randint(-18, 18)
                c = tuple(int(max(0, min(255, v + jitter))) for v in c)
                draw.rectangle([x1, y1, x2, y2], fill=c + (255,), outline=(0, 0, 0, 255), width=2)

            part_damage = {p: "none" for p in PARTS}
            damages = []
            k = random.randint(2, 3) if random.random() < multi_damage_ratio else 1
            damaged_parts = random.sample(PARTS, min(k, len(PARTS)))

            for p in damaged_parts:
                dtype = random.choice(DAMAGE_TYPES[1:])
                part_damage[p] = dtype
                if dtype == "missing":
                    self._apply_missing(img, parts[p])
                else:
                    # 40% of the time let damage extend past part edge (realistic: cracks at boundary)
                    extend = random.randint(8, 22) if random.random() < 0.4 else 0
                    box = self._subbox(parts[p], extend=extend, img_bounds=(W, H))
                    if dtype == "cracked":
                        self._draw_cracks(img, box)
                    elif dtype == "scratched":
                        self._draw_scratches(img, box)
                    elif dtype == "broken":
                        img = self._draw_broken(img, box)
                    elif dtype == "loose":
                        x1, y1, x2, y2 = parts[p]
                        shiftx = random.randint(-6, 6)
                        shifty = random.randint(-6, 6)
                        parts[p] = [
                            max(0, x1 + shiftx), max(0, y1 + shifty),
                            min(W - 1, x2 + shiftx), min(H - 1, y2 + shifty),
                        ]
                damages.append({"part": p, "type": dtype})

            # Domain randomization: rotation (±5°), perspective, shading, shadow, occlusion
            # Bboxes are updated to match so labels stay correct.
            if random.random() < 0.5:
                angle = random.uniform(-5, 5)
                cx, cy = W / 2, H / 2
                try:
                    rotated = img.rotate(
                        -angle,
                        center=(cx, cy),
                        expand=False,
                        resample=Image.BICUBIC,
                        fillcolor=(180, 180, 180),
                    )
                except TypeError:
                    try:
                        rotated = img.rotate(
                            -angle,
                            center=(cx, cy),
                            expand=False,
                            resample=Image.BICUBIC,
                            fill=(180, 180, 180),
                        )
                    except Exception:
                        rotated = None
                except Exception:
                    rotated = None
                if rotated is not None:
                    img = rotated
                    parts = self._rotate_boxes(parts, W, H, angle)
            if random.random() < 0.35:
                img, parts, W, H = self._apply_perspective(img, parts, W, H)
                assert img is not None, "_apply_perspective returned None image"
            if random.random() < 0.4:
                img = self._apply_depth_shading(img)
                assert img is not None, "_apply_depth_shading returned None"
            if random.random() < 0.3:
                img = self._apply_shadow(img, W, H)
                assert img is not None, "_apply_shadow returned None"
            if random.random() < 0.25:
                img = self._apply_occlusion(img, parts, W, H)
                assert img is not None, "_apply_occlusion returned None"

            if random.random() < 0.35:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
            if random.random() < 0.25:
                arr = np.array(img.convert("RGB"), dtype=np.int16)
                arr = np.clip(arr + np.random.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGBA")

            if img is None:
                raise RuntimeError(
                    "img became None during generation; check rotate, _apply_perspective, "
                    "_apply_depth_shading, _apply_shadow, _apply_occlusion (e.g. .paste() returns None)"
                )

            fname = f"img_{i:06d}.jpg"
            img.convert("RGB").save(os.path.join(self.img_dir, fname), quality=random.randint(80, 95))
            anns.append({
                "filename": fname,
                "width": W,
                "height": H,
                "parts": parts,
                "part_damage": part_damage,
                "damages": damages,
            })

        random.Random(seed).shuffle(anns)
        nval = int(len(anns) * val_ratio)
        val = anns[:nval]
        train = anns[nval:]
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "annotations_train.json"), "w", encoding="utf-8") as f:
            json.dump(train, f, indent=2)
        with open(os.path.join(self.output_dir, "annotations_val.json"), "w", encoding="utf-8") as f:
            json.dump(val, f, indent=2)
        with open(os.path.join(self.output_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump(anns, f, indent=2)
        print(f"[OK] train={len(train)} val={len(val)} images={len(anns)} at {self.output_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="./data/synth_v2")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--multi_damage_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    SyntheticChairGenV2(args.output_dir).generate(
        N=args.samples,
        val_ratio=args.val_ratio,
        multi_damage_ratio=args.multi_damage_ratio,
        seed=args.seed,
    )
