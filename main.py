##add the pi cameta and the model path

from scripts.detect_damage import detect_parts
from scripts.generate_repair_plan import generate_repair_plan
from scripts.render_visual_guidance import render_step_visual

def main():
    image_path = "./data/synthetic_damage/chair_missing_leg.png"
    model_path = "./models/damage_detection/part_detector.pth"
    
    detect_parts(image_path, model_path)

    # Example Manual Info
    furniture_type = "Chair"
    damaged_part = "Front Left Leg"
    assembly_step = "Attach the front left leg to the seat with screws."

    plan = generate_repair_plan(furniture_type, damaged_part, assembly_step)
    print("Repair Steps:\n", plan)

    # Render visual guide for the first step
    render_step_visual(
        "./data/partnet_data/chair/model.obj",
        highlighted_part_idx=2,
        save_path="./data/visual_guides/step1.png"
    )

if __name__ == "__main__":
    main()
