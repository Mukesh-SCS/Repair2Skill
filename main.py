"""
================================================================================
DESCRIPTION:
    End-to-end pipeline:
      capture/upload → detect → OpenAI repair plans → visual guides → graph.
    Also supports training:
      --train        → MobileNet classifier (train_part_detector.py)
      --train-frcnn  → Faster R-CNN detector  (train_detector_frcnn.py)

USAGE:
    # training
    python main.py --train
    python main.py --train-frcnn

    # data
    python main.py --generate-data --samples 2000

    # inference
    python main.py --upload ./data/user_images/chair.jpg
    python main.py --camera

OUTPUTS:
    ./outputs/stage1_parts.json
    ./outputs/repair_plan_<part>_<damage>.json
    ./data/visual_guides/<part>_repair_guide.png
    ./outputs/stage2_assembly_graph.json
Author Info: Mukesh Mani Tripathi
================================================================================
"""
import argparse, os, json, sys, subprocess
from pathlib import Path
from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect_damage_and_parts
from scripts.render_visual_guidance import render_step_visual
from scripts.generate_synthetic_data import SyntheticDataGenerator
from scripts.train_part_detector import train_model as train_classifier
from utils.openai_utils import generate_repair_plan
from utils.assembly_plan_utils import parse_manual

PARTS = ["seat","back","front_left_leg","front_right_leg",
         "back_left_leg","back_right_leg","armrest_left","armrest_right"]

def run_frcnn_training():
    subprocess.run([sys.executable, "scripts/train_detector_frcnn.py"], check=True)

def main():
    ap = argparse.ArgumentParser(description="Repair2Skill")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-frcnn", action="store_true")
    ap.add_argument("--generate-data", action="store_true")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--upload", type=str)
    args = ap.parse_args()

    if args.generate_data:
        SyntheticDataGenerator().generate_dataset(num_samples=args.samples); return
    if args.train:
        train_classifier(); return
    if args.train_frcnn:
        run_frcnn_training(); return

    # inference
    if args.camera and args.upload:
        raise ValueError("Use either --camera or --upload.")
    image_path = capture_from_camera() if args.camera else args.upload
    if not image_path: raise ValueError("Provide --camera or --upload.")
    print(f"Image: {image_path}")

    model_path = "./models/damage_detection/part_detector.pth"
    if not os.path.exists(model_path):
        print("Model not found. Train first with --train."); return

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Stage-I detection
    stage1 = detect_damage_and_parts(image_path, model_path=model_path)
    with open("outputs/stage1_parts.json", "w") as f: json.dump(stage1, f, indent=2)
    print("Saved outputs/stage1_parts.json")

    pairs = stage1.get("detected_pairs", [])
    if not pairs:
        print("No confident part↔damage pair; skipping plan and visuals."); return

    # OpenAI plans (uses OPENAI_API_KEY if set; otherwise fallback in openai_utils)
    for dp in pairs:
        part, dmg = dp["part"], dp["damage_type"]
        plan = generate_repair_plan("Chair", part, f"Repair the {part} that is {dmg}", dmg)
        out = f"outputs/repair_plan_{part}_{dmg}.json"
        with open(out, "w") as f: json.dump(plan, f, indent=2)
        print(f"Saved {out}")

    # Visuals for paired parts only
    Path("data/visual_guides").mkdir(parents=True, exist_ok=True)
    for dp in pairs:
        part = dp["part"]; idx = PARTS.index(part) if part in PARTS else -1
        render_step_visual(None, idx, f"data/visual_guides/{part}_repair_guide.png",
                   "outputs/stage1_parts.json", flip_horizontal=True, flip_vertical=True)


    # Stage-II graph from pairs
    graph = parse_manual(None, "outputs/stage1_parts.json")
    with open("outputs/stage2_assembly_graph.json", "w") as f: json.dump(graph, f, indent=2)
    print("Saved outputs/stage2_assembly_graph.json")

if __name__ == "__main__":
    main()
