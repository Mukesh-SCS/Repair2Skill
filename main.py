"""
================================================================================
DESCRIPTION:
    Main pipeline for the Repair2Skill project.
    - Loads or captures a chair image.
    - Runs the trained detector to identify damaged parts.
    - Uses the OpenAI API (if key present) to generate structured repair plans.
    - Renders visual guidance images for each targeted part.
    - Builds a deterministic assembly graph from detections.
    - Optional: trains baseline classifier (--train) or Faster R-CNN (--train-frcnn).

USAGE:
    # data + training
    python main.py --generate-data --samples 2000
    python main.py --train
    python main.py --train-frcnn --frcnn-ann ./data/synthetic_damage/annotations.json \
                   --frcnn-imgs ./data/synthetic_damage/images --frcnn-epochs 10 --frcnn-batch 2

    # inference
    python main.py --upload ./data/user_images/chair.jpg
    python main.py --camera
    # Optional: export OPENAI_API_KEY, OPENAI_MODEL=gpt-4o-mini

OUTPUTS:
    ./outputs/stage1_parts.json
    ./outputs/repair_plan_<part>_<damage>.json
    ./data/visual_guides/*.png
    ./outputs/stage2_assembly_graph.json

ARGUMENTS:
    --camera                    Capture image from camera
    --upload PATH               Path to an image
    --generate-data             Generate synthetic dataset
    --samples INT               Samples for synthetic generation (default 1000)
    --train                     Train baseline MobileNet classifier
    --train-frcnn               Train Faster R-CNN detector (runs separate script)
    --frcnn-ann PATH            Annotations JSON for FRCNN
    --frcnn-imgs PATH           Images folder for FRCNN
    --frcnn-epochs INT          FRCNN epochs (default 10)
    --frcnn-batch INT           FRCNN batch size (default 2)
Author Info: Mukesh Mani Tripathi
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
from scripts.generate_synthetic_data import SyntheticDataGenerator
from scripts.train_part_detector import train_model
from utils.openai_utils import generate_repair_plan
from utils.assembly_plan_utils import parse_manual





def main():
    ap = argparse.ArgumentParser(description="Repair2Skill: detect, plan, visualize")
    # data + training
    ap.add_argument("--generate-data", action="store_true", help="Generate synthetic dataset")
    ap.add_argument("--samples", type=int, default=1000, help="Synthetic samples to generate")
    ap.add_argument("--train", action="store_true", help="Train baseline MobileNet classifier")
    ap.add_argument("--train-frcnn", action="store_true", help="Train Faster R-CNN detector via script")
    ap.add_argument("--frcnn-ann", type=str, default="./data/synthetic_damage/annotations.json")
    ap.add_argument("--frcnn-imgs", type=str, default="./data/synthetic_damage/images")
    ap.add_argument("--frcnn-epochs", type=int, default=10)
    ap.add_argument("--frcnn-batch", type=int, default=2)

    # inference
    ap.add_argument("--camera", action="store_true", help="Capture image via camera")
    ap.add_argument("--upload", type=str, help="Path to an image file")
    args = ap.parse_args()

    # Generate synthetic data
    if args.generate_data:
        print("Generating synthetic dataset...")
        SyntheticDataGenerator().generate_dataset(num_samples=args.samples)
        print("Done.")
        return

    # Train baseline classifier
    if args.train:
        print("Training damage/part classifier baseline...")
        train_model()
        print("Training complete.")
        return

    # Train Faster R-CNN detector
    if args.train_frcnn:
        run_frcnn_training(args.frcnn_ann, args.frcnn_imgs, args.frcnn_epochs, args.frcnn_batch)
        return

    # Inference path requires one image source
    if args.camera and args.upload:
        raise ValueError("Use either --camera or --upload, not both.")
    if args.camera:
        image_path = capture_from_camera()
    elif args.upload:
        image_path = args.upload
    else:
        raise ValueError("Provide --camera or --upload.")
    print(f"Image: {image_path}")

    # Check classifier model
    model_path = "./models/damage_detection/part_detector.pth"
    if not os.path.exists(model_path):
        print("Model not found. Train first with --train (or generate data then train).")
        return

    # Stage I: detection
    Path("outputs").mkdir(parents=True, exist_ok=True)
    print("Detecting damage and parts...")
    stage1 = detect_damage_and_parts(image_path, model_path=model_path)
    print(json.dumps(stage1, indent=2))
    with open("outputs/stage1_parts.json", "w") as f:
        json.dump(stage1, f, indent=2)

    # Build simple pairs if missing
    pairs = stage1.get("detected_pairs", [])
    if not pairs and stage1.get("detected_damages") and stage1.get("detected_parts"):
        dmg = stage1["detected_damages"][0]["type"]
        pairs = [{
            "part": p["part"],
            "damage_type": dmg,
            "damage_confidence": 0.5,
            "part_confidence": p["confidence"]
        } for p in stage1["detected_parts"]]

    # OpenAI repair plans (uses OPENAI_API_KEY if set; otherwise returns local fallback)
    if pairs:
        print("\nGenerating repair plans...")
        for dp in pairs:
            part = dp["part"]; dmg = dp["damage_type"]
            assembly_step = f"Repair the {part} that is {dmg}"
            plan = generate_repair_plan("Chair", part, assembly_step, dmg)
            out = f"outputs/repair_plan_{part}_{dmg}.json"
            with open(out, "w") as f:
                json.dump(plan, f, indent=2)
            print(f"Saved {out}")
    else:
        print("No pairs found. Skipping plan generation.")

    # Visual guidance images
    Path("data/visual_guides").mkdir(parents=True, exist_ok=True)
    targeted_parts = [p["part"] for p in pairs] if pairs else [p["part"] for p in stage1.get("detected_parts", [])]
    PARTS = ["seat","back","front_left_leg","front_right_leg","back_left_leg","back_right_leg","armrest_left","armrest_right"]
    for part in targeted_parts:
        try:
            idx = PARTS.index(part)
        except ValueError:
            idx = -1
        render_step_visual(
            model_path=None,
            highlighted_part_idx=idx,
            save_path=f"data/visual_guides/{part}_repair_guide.png",
            damage_report_path="outputs/stage1_parts.json"
        )

    # Stage II: simple rule-based assembly graph
    print("\nGenerating assembly graph...")
    graph = parse_manual(None, "outputs/stage1_parts.json")
    with open("outputs/stage2_assembly_graph.json", "w") as f:
        json.dump(graph, f, indent=2)
    print("Saved outputs/stage2_assembly_graph.json")

    print("\nDone. See ./outputs and ./data/visual_guides")


def run_frcnn_training(ann: str, imgs: str, epochs: int, batch: int):
    """Calls scripts/train_detector_frcnn.py in a subprocess with provided args."""
    cmd = [
        sys.executable, "scripts/train_detector_frcnn.py",
        "--ann", ann,
        "--imgs", imgs,
        "--epochs", str(epochs),
        "--batch", str(batch),
    ]
    print("Launching Faster R-CNN training:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
