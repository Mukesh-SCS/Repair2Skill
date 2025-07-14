# ==================== scripts/generate_synthetic_data.py ====================import os
import os 
import json
import random
from PIL import Image, ImageDraw
from tqdm import tqdm

def generate_image(index):
    img = Image.new('RGB', (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    part_labels = []
    damage_labels = []
    for part in PART_CLASSES:
        if random.random() > 0.3:
            draw.rectangle([random.randint(10, 100), random.randint(10, 100), random.randint(120, 200), random.randint(120, 200)], fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            part_labels.append(part)
            if random.random() > 0.6:
                damage = random.choice(DAMAGE_CLASSES)
                damage_labels.append(damage)

    image_path = f"data/synthetic/images/image_{index:04d}.jpg"
    label_path = f"data/synthetic/labels/label_{index:04d}.json"

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    img.save(image_path)
    with open(label_path, 'w') as f:
        json.dump({"parts": part_labels, "damages": damage_labels}, f)

def main(samples):
    for i in tqdm(range(samples)):
        generate_image(i)
    print(f"Generated {samples} synthetic images with annotations")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()
    main(args.samples)

# --- main.py ---
import os
import argparse
from detect_damage import detect_damage_and_parts

def save_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def main(image_path):
    print(f"Using uploaded image: {image_path}")
    print("Detecting damage and parts...")

    model_path = "./models/damage_detection/part_detector.pth"
    detection = detect_damage_and_parts(image_path, model_path)
    save_json(detection, "outputs/stage1_parts.json")

    print("Stage I Output (Detected Parts JSON):")
    print(json.dumps(detection, indent=2))

    # Simulate next steps or plug into rendering/graph logic
    print("\nRepair analysis complete!")
    print("Check the ./outputs/ directory for results:")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', type=str, help='Path to uploaded image')
    args = parser.parse_args()
    main(args.upload)