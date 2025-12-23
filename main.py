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

import argparse, os, json, subprocess, sys
from pathlib import Path
from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect as detect_damage_and_parts
from scripts.generate_repair_plan import generate_repair_plan
from scripts.repair_graph import generate_repair_graph, save_repair_graph_json, visualize_repair_graph
from scripts.render_visual_guidance import render_step_visual
from scripts.generate_synthetic_data import SyntheticDataGenerator


def main():
    ap = argparse.ArgumentParser(description="Repair2Skill unified pipeline")
    ap.add_argument("--generate-data", action="store_true")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--train-frcnn", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--upload", type=str)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--debug", action="store_true", help="Enable debug mode for detection")
    args = ap.parse_args()

    # ---- Stage 0: Data Gen / Training ----
    if args.generate_data:
        SyntheticDataGenerator().generate_dataset(N=args.samples)
        return

    if args.train_frcnn:
        subprocess.run([
            sys.executable, "scripts/train_detector_mobilenet.py",
            "--epochs", str(args.epochs),
            "--batch", str(args.batch)
        ], check=True)
        return

    # ---- Stage 1: Image Input ----
    if args.camera and args.upload:
        raise ValueError("Use either --camera or --upload, not both.")

    image_path = capture_from_camera() if args.camera else args.upload
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[INFO] Using image: {image_path}")

    model_path = "./models/damage_detection/mobilenet_ssd.pth"
    if not os.path.exists(model_path):
        print("[ERROR] Model not found. Train first.")
        return

    os.makedirs("outputs", exist_ok=True)

    # ---- Stage 2: Detection ----
    stage1 = detect_damage_and_parts(image_path, weights=model_path, threshold=args.threshold, debug=args.debug)
    with open("outputs/stage1_parts.json", "w") as f:
        json.dump(stage1, f, indent=2)

    pairs = stage1.get("detected_pairs", [])
    if not pairs:
        print("[WARN] No damaged parts detected.")
        return

    # Pick top detected damage-part pair
    if pairs:
        # Use pre-calculated smart_score from detection (set in detect_damage.py)
        # If not present, fall back to old formula for compatibility
        scored_pairs = []
        for pair in pairs:
            if 'smart_score' in pair:
                score = pair['smart_score']
            else:
                # Fallback for old detection outputs
                overlap = pair.get("overlap_iou", 0.0)
                overlap_bonus = max(0.5, overlap) if overlap > 0.15 else 0.3
                score = pair["part_confidence"] * pair["damage_confidence"] * overlap_bonus
            scored_pairs.append((score, pair))
        
        scored_pairs.sort(key=lambda x: -x[0])
        dp = scored_pairs[0][1]
    else:
        dp = pairs[0] if pairs else None
    
    part, dmg = dp["part"], dp["damage_type"]
    print(f"[INFO] Top damage: {part} ({dmg})")

    # ---- Stage 3: LLM Plan ----
    plan = generate_repair_plan("Chair", part, dmg)
    if "repair_sequence" not in plan:
        print("[ERROR] Invalid repair plan from GPT.")
        return

    plan_path = f"outputs/repair_plan_{part}_{dmg}.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    # ---- Stage 4: Graph ----
    graph = generate_repair_graph(part)
    graph_path = save_repair_graph_json(graph, part)
    visualize_repair_graph(graph, f"outputs/repair_graph_{part}.png")

    # ---- Stage 5: Visual Guide ----
    out_img = f"data/visual_guides/{part}_repair_guide.png"
    os.makedirs("data/visual_guides", exist_ok=True)

    render_step_visual(
        model_path=None,
        highlighted_part_idx=None,
        save_path=out_img,
        damage_report_path="outputs/stage1_parts.json",
        plan_json_path=plan_path,
    )



if __name__ == "__main__":
    main()
