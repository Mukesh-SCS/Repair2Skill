"""
================================================================================
Repair2Skill Main Pipeline
================================================================================
Stages:
  1. Capture/Upload Image → Faster R-CNN Detection
  2. GPT-4o Repair Plan Generation
  3. Visual Repair Guide
  4. Repair Dependency Graph
  5. (Next) Robotic Execution in PyBullet

Usage:
  # Data generation
  python main.py --generate-data --samples 1000

  # Training
  python main.py --train-frcnn

  # Inference
  python main.py --camera
  python main.py --upload ./data/user_images/chair.jpg
================================================================================
"""

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect_damage_and_parts
from scripts.render_visual_guidance import render_step_visual
from utils.openai_utils import generate_repair_plan
from scripts.repair_graph import generate_repair_graph, save_repair_graph_json, visualize_repair_graph


# -----------------------------------------------------------------------------


def run_frcnn_training():
    """Launch training script for Faster R-CNN detector."""
    subprocess.run([sys.executable, "scripts/train_detector_frcnn.py"], check=True)


def main():
    ap = argparse.ArgumentParser(description="Repair2Skill pipeline")
    ap.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    ap.add_argument("--samples", type=int, default=1000, help="Synthetic sample count")
    ap.add_argument("--train-frcnn", action="store_true", help="Train Faster R-CNN detector")
    ap.add_argument("--camera", action="store_true", help="Capture from Pi Camera/Webcam")
    ap.add_argument("--upload", type=str, help="Path to input image")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Stage 0: Data Generation / Training
    # -------------------------------------------------------------------------
    if args.generate_data:
        from scripts.generate_synthetic_data import SyntheticDataGenerator
        SyntheticDataGenerator().generate_dataset(num_samples=args.samples)
        return

    if args.train_frcnn:
        run_frcnn_training()
        return

    # -------------------------------------------------------------------------
    # Stage 1: Image Capture or Upload
    # -------------------------------------------------------------------------
    if args.camera and args.upload:
        raise ValueError("Use either --camera or --upload, not both.")

    image_path = capture_from_camera() if args.camera else args.upload
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError("Image path not found.")

    print(f"[INFO] Using image: {image_path}")

    model_path = "./models/damage_detection/frcnn_model.pth"
    if not os.path.exists(model_path):
        print("[ERROR] Model not found. Train first with --train-frcnn.")
        return

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Stage 2: Damage Detection (Faster R-CNN)
    # -------------------------------------------------------------------------
    stage1 = detect_damage_and_parts(image_path, model_path=model_path)
    with open("outputs/stage1_parts.json", "w") as f:
        json.dump(stage1, f, indent=2)
    print("[OK] Saved outputs/stage1_parts.json")

    pairs = stage1.get("detected_pairs", [])
    if not pairs:
        print("[WARN] No confident part-damage pair found.")
        return

    # -------------------------------------------------------------------------
    # Stage 3: GPT-4o Repair Plan
    # -------------------------------------------------------------------------
    for dp in pairs:
        part, dmg = dp["part"], dp["damage_type"]
        plan = generate_repair_plan("Chair", part, dmg)
        out_path = f"outputs/repair_plan_{part}_{dmg}.json"
        with open(out_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"[OK] Saved {out_path}")

        # ---------------------------------------------------------------------
        # Stage 4: Repair Dependency Graph (Manual2Skill-style)
        # ---------------------------------------------------------------------
        graph = generate_repair_graph(part)
        graph_path = save_repair_graph_json(graph, part)
        visualize_repair_graph(graph, f"outputs/repair_graph_{part}.png")

        # ---------------------------------------------------------------------
        # Stage 5: Visual Repair Guide
        # ---------------------------------------------------------------------
        Path("data/visual_guides").mkdir(parents=True, exist_ok=True)
        out_img = f"data/visual_guides/{part}_repair_guide.png"
        render_step_visual(
            model_path=None,
            highlighted_part_idx=None,
            save_path=out_img,
            damage_report_path="outputs/stage1_parts.json",
            plan_json_path=out_path,
            mirror_horizontal=True,
        )

    print("[INFO] Repair2Skill pipeline completed successfully.")


if __name__ == "__main__":
    main()
