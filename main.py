import argparse
from scripts.capture_image import capture_from_camera
from scripts.detect_damage import detect_damage_from_image # Assuming this function is defined in detect_damage.py
from scripts.detect_damage import detect_parts
from scripts.generate_repair_plan import generate_repair_plan
from scripts.render_visual_guidance import render_step_visual

def main():
    image_path = "./data/synthetic_damage/chair_missing_leg.png"
    model_path = "./models/damage_detection/part_detector.pth"
    
    detect_parts(image_path, model_path)
    
    parser = argparse.ArgumentParser(description='Furniture Repair Model')
    parser.add_argument('--camera', action='store_true', help='Capture image using Pi Camera')
    parser.add_argument('--upload', type=str, help='Path to the user-uploaded image')
    args = parser.parse_args()
     
    if args.camera:
       image_path = capture_from_camera()
       print(f"Image captured: {image_path}")
    elif args.upload:
        image_path = args.upload
        print(f"Using uploaded image: {image_path}")
    else:
        raise ValueError("Please provide --camera or --upload argument.")

    # Detect damage from the image
    damage_report = detect_damage_from_image(image_path)    
    print("Damage Report:", damage_report)
    # Proceed with further pipeline (repair planning, visualization)
    # Generate repair plan based on detected damage
    # Assuming the damage_report contains necessary information to generate the plan    
    from utils.openai_utils import generate_repair_plan
    # Example usage of generate_repair_plan
    # This function should be defined in utils/openai_utils.py
    # The function should take furniture type, damaged part, and assembly step as inputs
    # and return a step-by-step repair guide.
    # Example inputs for the function
    # Note: The actual inputs should be derived from the damage_report
    # For demonstration, we will use hardcoded values.
    
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


