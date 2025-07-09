# ==================== main.py ====================
import argparse
import os
import json
from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect_damage_from_image, detect_parts
from scripts.render_visual_guidance import render_step_visual
from scripts.generate_synthetic_data import SyntheticDataGenerator
from scripts.train_part_detector import train_model
from utils.openai_utils import generate_repair_plan
from utils.assembly_plan_utils import parse_manual

def main():
    parser = argparse.ArgumentParser(description='Furniture Repair Model')
    parser.add_argument('--camera', action='store_true', help='Capture image using Pi Camera')
    parser.add_argument('--upload', type=str, help='Path to the user-uploaded image')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic training data')
    parser.add_argument('--train', action='store_true', help='Train the damage detection model')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples to generate')
    parser.add_argument('--ikea-parse', action='store_true', help='Parse IKEA manual dataset')
    parser.add_argument('--summarize', action='store_true', help='Summarize IKEA part usage')

    
    args = parser.parse_args()
    
    

    # Handle IKEA manual parsing
    if args.ikea_parse:
        from scripts.ikea_parser import parse_main_data_json, summarize_parts
        ikea_json_path = "./data/ikea_manuals/main_data.json"
        output_path = "./outputs/ikea_assembly_graph.json"
        print(f"Parsing IKEA manual from {ikea_json_path}...")
        parse_main_data_json(ikea_json_path, output_path)
        if args.summarize:
            summarize_parts(output_path)
        print(f"IKEA assembly graph saved to {output_path}")
        return

    # Generate synthetic data if requested
    if args.generate_data:
        print("Generating synthetic training data...")
        generator = SyntheticDataGenerator()
        generator.generate_dataset(num_samples=args.samples)
        print("Synthetic data generation complete!")
        return
    
    # Train model if requested
    if args.train:
        print("Training damage detection model...")
        train_model()
        print("Model training complete!")
        return
    
    # Image capture/upload logic
    if args.camera:
        image_path = capture_from_camera()
        print(f"Image captured: {image_path}")
    elif args.upload:
        image_path = args.upload
        print(f"Using uploaded image: {image_path}")
    else:
        raise ValueError("Please provide --camera or --upload argument.")

    # Check if model exists
    model_path = "./models/damage_detection/part_detector.pth"
    if not os.path.exists(model_path):
        print("Model not found! Please train the model first using --train flag")
        print("Or generate synthetic data first using --generate-data flag")
        return

    # Detect damage from the image (Stage I output)
    print("Detecting damage and parts...")
    damage_report = detect_damage_from_image(image_path)
    print("Stage I Output (Detected Parts JSON):")
    print(json.dumps(damage_report, indent=2))

    # Save Stage I output
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/stage1_parts.json", "w") as f:
        json.dump(damage_report, f, indent=2)

    # Generate repair plan for each detected damage
    if "detected_damages" in damage_report and damage_report["detected_damages"]:
        print("\nGenerating repair plans...")
        
        for i, damage in enumerate(damage_report["detected_damages"]):
            # Find corresponding part
            furniture_type = "Chair"
            damaged_part = "Unknown"
            
            # Try to match damage with detected parts
            if "detected_parts" in damage_report and damage_report["detected_parts"]:
                if i < len(damage_report["detected_parts"]):
                    damaged_part = damage_report["detected_parts"][i]["part"]
            
            damage_type = damage["type"]
            assembly_step = f"Repair the {damaged_part} that is {damage_type}"
            
            print(f"\nGenerating repair plan for {damaged_part} ({damage_type})...")
            plan = generate_repair_plan(furniture_type, damaged_part, assembly_step, damage_type)
            
            # Save individual repair plan
            plan_filename = f"./outputs/repair_plan_{damaged_part}_{damage_type}.json"
            with open(plan_filename, "w") as f:
                json.dump(plan, f, indent=2)
            
            print(f"Repair plan saved to: {plan_filename}")
    else:
        print("No damage detected in the image.")

    # Ensure output directory exists
    os.makedirs("./data/visual_guides", exist_ok=True)

    # Render visual guide for detected parts
    if "detected_parts" in damage_report and damage_report["detected_parts"]:
        for i, part in enumerate(damage_report["detected_parts"]):
            part_name = part["part"]
            confidence = part["confidence"]
            
            print(f"Rendering visual guide for {part_name} (confidence: {confidence:.2f})...")
            render_step_visual(
                "./data/partnet_data/chair/model.obj",
                highlighted_part_idx=i,
                save_path=f"./data/visual_guides/{part_name}_repair_guide.png"
            )

    # Stage II: Hierarchical Graph Generation
    print("\nGenerating assembly graph...")
    assembly_graph = parse_manual("./data/furniture_manuals/sample_manual.png", "./outputs/stage1_parts.json")
    print("Stage II Output (Assembly Graph):")
    print(json.dumps(assembly_graph, indent=2))
    
    # Save assembly graph
    with open("./outputs/stage2_assembly_graph.json", "w") as f:
        json.dump(assembly_graph, f, indent=2)
    
    print("\nRepair analysis complete!")
    print("Check the ./outputs/ directory for results:")
    print("- stage1_parts.json: Detected parts and damages")
    print("- stage2_assembly_graph.json: Repair sequence graph")
    print("- repair_plan_*.json: Individual repair plans")
    print("- ./data/visual_guides/: Visual repair guides")

if __name__ == "__main__":
    main()