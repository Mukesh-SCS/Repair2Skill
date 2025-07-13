
# 🪑 Repair2Skill Setup and Model Training Guide (Windows)

This guide walks you through setting up your environment and training the damage detection model to generate `part_detector.pth`.

---

## ✅ Step 1: Set Up Python Environment (Windows)

### 1. Install Python 3.10 or 3.9
Download from: https://www.python.org/downloads/

> ⚠️ Do NOT use Python 3.11+ (Detectron2 is not supported)

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies
Make sure your `requirements.txt` looks like this (no detectron2 issues):

```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.21.0
matplotlib>=3.3.0
openai>=1.0.0
python-dotenv>=0.19.0
requests>=2.25.0
json5>=0.9.0
tqdm>=4.62.0
albumentations>=1.0.0
pycocotools>=2.0.0
transformers>=4.20.0
# Optional: Raspberry Pi only
picamera2>=0.3.0; platform_system == "Linux"
open3d>=0.13.0
trimesh>=3.9.0
```

Then install:
```bash
pip install -r requirements.txt
```

---

## ✅ Step 2: Train the Damage Detection Model

### 1. Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py --samples 500
```

This saves images and `annotations.json` to `./data/synthetic_damage/`.

### 2. Train the Model
```bash
python scripts/train_part_detector.py --data-dir ./data/synthetic_damage/
```

After training, your model will be saved to:
```
./models/damage_detection/part_detector.pth
```

---

## ✅ Step 3: Set OpenAI API Key

Create a `.env` file in your project root with:
```env
OPENAI_API_KEY=sk-...
```

---

## ✅ Step 4: Run the Pipeline

### 1. Place your test image
Example:
```
./data/user_images/my_broken_chair.jpg
```

### 2. Run:
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```

---

## 🔍 What Happens Internally

1. Loads your image
2. Uses the trained model (`part_detector.pth`) to detect parts
3. Sends it to OpenAI GPT-4o via `analyze_image`
4. Outputs JSON file `./outputs/stage1_parts.json`

---

## 🧪 Output Example

Console:
```json
[{"name": "seat", "label": 0, "role": "support"}, {"name": "leg", "label": 1, "role": "support"}]
```

File:
```
./outputs/stage1_parts.json
```

---

## 🛠 Troubleshooting

- **Model Not Found:** Make sure `part_detector.pth` exists in `./models/damage_detection/`
- **OpenAI Error:** Double check your `.env` key
- **Image Format:** Use `.jpg` or `.png`

---

## ✅ Done!

You're ready to detect broken chair parts and build repair plans.





__________________________________________________________________________________
# 🪑 Repair2Skill Setup and Training Guide (Raspberry Pi 5)

This guide helps you set up the Repair2Skill pipeline and train a part damage detector on Raspberry Pi 5 using synthetic or real image data.

---

## ✅ Step 1: Set Up Python & Virtual Environment

### 1. Install Python 3.9 (recommended for compatibility)

```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

### 2. Create and Activate Virtual Environment

```bash
python3.9 -m venv venv
source venv/bin/activate
```

---

## ✅ Step 2: Install Dependencies

Ensure your `requirements.txt` includes this (Pi-compatible):

```txt
torch==2.0.1
torchvision==0.15.2
opencv-python-headless>=4.5.0
pillow>=8.0.0
numpy>=1.21.0
matplotlib>=3.3.0
openai>=1.0.0
python-dotenv>=0.19.0
requests>=2.25.0
json5>=0.9.0
tqdm>=4.62.0
albumentations>=1.0.0
pycocotools>=2.0.0
transformers>=4.20.0
picamera2>=0.3.0
open3d>=0.13.0
trimesh>=3.9.0
```

> ⚠️ Use `opencv-python-headless` instead of `opencv-python` to avoid GUI issues on Pi.

Install with:
```bash
pip install -r requirements.txt
```

---

## ✅ Step 3: (Optional) Capture Real Images with Pi Camera

Use the GUI tool:

```bash
python scripts/real_data_collector.py
```

- Capture damaged parts using PiCamera2
- Annotate images and save bounding boxes

---

## ✅ Step 4: Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --samples 500
```

This will output:
```
./data/synthetic_damage/images/
./data/synthetic_damage/annotations.json
```

---

## ✅ Step 5: Train the Part Detector Model

```bash
python scripts/train_part_detector.py --data-dir ./data/synthetic_damage/
```

Output model:
```
./models/damage_detection/part_detector.pth
```

---

## ✅ Step 6: Set OpenAI API Key

Create a `.env` file in your project root with:

```env
OPENAI_API_KEY=sk-...
```

---

## ✅ Step 7: Run Full Detection Pipeline

Put a test image here:
```
./data/user_images/my_broken_chair.jpg
```

Then run:
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```

---

## 🛠 Troubleshooting (Pi)

- **Camera errors?** Make sure `libcamera` is working and `picamera2` is installed
- **Slow training?** Use fewer samples like `--samples 200`
- **Torch issues?** Use pre-built wheels for ARM64 from [https://github.com/nmilosev/pytorch-arm-builds](https://github.com/nmilosev/pytorch-arm-builds)

---



### DElete after use ##

Quick Start: Furniture Repair Model Pipeline
1. Prerequisites
Raspberry Pi 5 with 64-bit OS.
Pi Camera connected and working.
3D model at ./data/partnet_data/chair/model.obj.
Texture files in ./data/textures/.
OpenAI API key in .env as OPENAI_API_KEY=your-api-key-here.
Virtual environment activated:

source venv/bin/activate
All code and config files are up to date.
2. Generate Synthetic Training Data

python scripts/generate_synthetic_data.py --samples 1000
Check: ./data/synthetic_damage/images/ for images, and annotation files for content.
3. Train the Detection Model

python scripts/train_detector_frcnn.py
Check: ./models/damage_detection/frcnn_model.pth is created.
Tip: If you get memory errors, reduce batch size in model_config.yaml.
4. (Optional) Optimize the Model

python scripts/model_optimization.py
Check: quantized_model.pth and pruned_model.pth in damage_detection.
5. Test the Pipeline
Capture an image (optional):

python scripts/capture_image.py
Run the main script with your image:

python main.py --upload ./data/user_images/your_image.jpg
Check outputs:
./outputs/stage1_parts.json
./outputs/repair_plan_*.json
./outputs/stage2_assembly_graph.json
visual_guides
6. Evaluate the Model

python scripts/evaluate_model.py --test-data-path ./data/synthetic_damage/
Check: ./outputs/evaluation_results.png and .json for metrics.
Troubleshooting
No damage detected? Lower threshold in detect_damage.py.
Visual guide incorrect? Check damage_report_path in main.py.
Slow or errors? Reduce --samples or batch size.
You can copy and follow these steps directly.
Let me know if you want a printable version or further simplification!
