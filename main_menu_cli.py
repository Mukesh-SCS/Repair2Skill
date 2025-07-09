
import os
import json
from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect_damage_from_image
from scripts.render_visual_guidance import render_step_visual
from scripts.generate_synthetic_data import SyntheticDataGenerator
from scripts.train_part_detector import train_model
from scripts.evaluate_model import evaluate_model
from scripts.data_augmentation import AdvancedDataAugmentation
from scripts.real_data_collector import RealDataCollector
from scripts.model_optimization import ModelOptimizer
from utils.openai_utils import generate_repair_plan
from utils.assembly_plan_utils import parse_manual
from scripts.ikea_parser import parse_main_data_json, summarize_parts

def menu():
    while True:
        print("""
=========== Repair2Skill - Main Menu ===========
1. Capture image from camera
2. Upload existing image
3. Generate synthetic training data
4. Train damage detection model
5. Detect damage from image
6. Generate IKEA assembly graph
7. Evaluate trained model
8. Augment dataset
9. Collect real data with labels
10. Optimize trained model
0. Exit
""")
        choice = input("Select an option: ").strip()

        if choice == "1":
            image_path = capture_from_camera()
            handle_image_analysis(image_path)

        elif choice == "2":
            image_path = input("Enter path to uploaded image: ").strip()
            if os.path.exists(image_path):
                handle_image_analysis(image_path)
            else:
                print("❌ File not found.")

        elif choice == "3":
            samples = int(input("Enter number of synthetic samples to generate: "))
            SyntheticDataGenerator().generate_dataset(num_samples=samples)

        elif choice == "4":
            train_model()

        elif choice == "5":
            image_path = input("Enter path to image: ").strip()
            handle_image_analysis(image_path)

        elif choice == "6":
            ikea_json_path = input("Enter IKEA main_data.json path: ").strip()
            output_path = "./outputs/ikea_assembly_graph.json"
            parse_main_data_json(ikea_json_path, output_path)
            summarize = input("Summarize part usage? (y/n): ").strip().lower()
            if summarize == "y":
                summarize_parts(output_path)

        elif choice == "7":
            model_path = input("Path to trained model: ").strip()
            data_path = input("Path to test dataset: ").strip()
            evaluate_model(model_path, data_path)

        elif choice == "8":
            input_dir = input("Path to existing dataset: ").strip()
            output_dir = input("Path to save augmented dataset: ").strip()
            multiplier = int(input("How many augmentations per image? "))
            AdvancedDataAugmentation().augment_dataset(input_dir, output_dir, multiplier)

        elif choice == "9":
            collector = RealDataCollector()
            n = int(input("How many images to collect? "))
            collector.batch_collect(num_images=n)

        elif choice == "10":
            model_path = input("Path to trained model: ").strip()
            optimizer = ModelOptimizer(model_path)
            optimizer.benchmark_models()

        elif choice == "0":
            print("✅ Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def handle_image_analysis(image_path):
    print(f"Analyzing image: {image_path}")
    model_path = "./models/damage_detection/part_detector.pth"
    if not os.path.exists(model_path):
        print("❌ Model not found. Train it first.")
        return

    damage_report = detect_damage_from_image(image_path)
    print(json.dumps(damage_report, indent=2))

    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/stage1_parts.json", "w") as f:
        json.dump(damage_report, f, indent=2)

    if "detected_damages" in damage_report:
        for i, damage in enumerate(damage_report["detected_damages"]):
            damaged_part = "unknown"
            if i < len(damage_report.get("detected_parts", [])):
                damaged_part = damage_report["detected_parts"][i]["part"]
            assembly_step = f"Repair {damaged_part} that is {damage['type']}"
            plan = generate_repair_plan("Chair", damaged_part, assembly_step, damage["type"])
            filename = f"./outputs/repair_plan_{damaged_part}_{damage['type']}.json"
            with open(filename, "w") as f:
                json.dump(plan, f, indent=2)
            print(f"🔧 Repair plan saved to {filename}")

    if "detected_parts" in damage_report:
        for i, part in enumerate(damage_report["detected_parts"]):
            render_step_visual(
                "./data/partnet_data/chair/model.obj",
                highlighted_part_idx=i,
                save_path=f"./data/visual_guides/{part['part']}_repair_guide.png"
            )

    # Hierarchical graph generation
    assembly_graph = parse_manual("./data/furniture_manuals/sample_manual.png", "./outputs/stage1_parts.json")
    with open("./outputs/stage2_assembly_graph.json", "w") as f:
        json.dump(assembly_graph, f, indent=2)
    print("📊 Assembly graph saved to ./outputs/stage2_assembly_graph.json")

if __name__ == "__main__":
    menu()
