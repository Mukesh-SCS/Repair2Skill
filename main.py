import argparse
import os
import json
# from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect_damage_from_image, detect_parts
from scripts.render_visual_guidance import render_step_visual
from utils.openai_utils import generate_repair_plan
from utils.assembly_plan_utils import parse_manual

def main():
    parser = argparse.ArgumentParser(description='Furniture Repair Model')
    parser.add_argument('--camera', action='store_true', help='Capture image using Pi Camera')
    parser.add_argument('--upload', type=str, help='Path to the user-uploaded image')
    args = parser.parse_args()

    if args.camera:
       # image_path = capture_from_camera()
        print(f"Image captured: {image_path}")
    elif args.upload:
        image_path = args.upload
        print(f"Using uploaded image: {image_path}")
    else:
        raise ValueError("Please provide --camera or --upload argument.")

    model_path = "./models/damage_detection/part_detector.pth"
    detect_parts(image_path, model_path)

    # Detect damage from the image (Stage I output)
    damage_report = detect_damage_from_image(image_path)
    print("Stage I Output (Detected Parts JSON):")
    print(json.dumps(damage_report, indent=2))

    # Save Stage I output
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/stage1_parts.json", "w") as f:
        json.dump(damage_report, f, indent=2)

    # Example Manual Info (should be parsed from damage_report in a real pipeline)
    furniture_type = "Chair"
    damaged_part = "Front Left Leg"
    assembly_step = "Attach the front left leg to the seat with screws."

    plan = generate_repair_plan(furniture_type, damaged_part, assembly_step)
    print("Repair Steps:\n", plan)

    # Ensure output directory exists
    os.makedirs("./data/visual_guides", exist_ok=True)

    # Render visual guide for the first step
    render_step_visual(
        "./data/partnet_data/chair/model.obj",
        highlighted_part_idx=2,
        save_path="./data/visual_guides/step1.png"
    )

    # Stage II: Hierarchical Graph Generation (placeholder)
    manual_path = "./data/furniture_manuals/sample_manual.png"  # Use PNG/JPG for VLM input
    assembly_graph = parse_manual(manual_path, "./outputs/stage1_parts.json")
    print("Stage II Output (Assembly Graph):")
    print(assembly_graph)
    # Optionally save to JSON
    with open("./outputs/stage2_assembly_graph.json", "w") as f:
        json.dump(assembly_graph, f, indent=2)

if __name__ == "__main__":
    main()


