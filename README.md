# 🛠️ Repair2Skill on Raspberry Pi 5

**Repair2Skill** is an AI-powered framework that enables robots (or intelligent agents) to detect and plan repairs for broken furniture parts. Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this project extends the original idea to handle **furniture repair** by using a camera, part detection, and **vision-language models (VLMs)** like GPT-4o.

This implementation runs on a **Raspberry Pi 5** equipped with a camera for real-time image capture.

---

## 🧠 What It Does

- Captures an image of a **broken chair** using a Pi Camera or uploaded image.
- Uses a trained **Faster R-CNN** or custom classifier to detect damaged parts.
- Sends detection results to **GPT-4o** via OpenAI API to generate a **step-by-step repair plan**.
- Generates a **visual repair graph** for robotic or manual execution.
- Optionally executes repair plans step-by-step via an executor.

---

## 📁 Project Structure
```bash
FurnitureRepairModel/
├── data/
│   ├── partnet_data/
│   ├── ikea_manuals/
│   ├── synthetic_damage/
│   └── user_images/
│
├── models/
│   ├── damage_detection/
│   ├── pose_estimation/
│   └── openai_integration/
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_part_detector.py
│   ├── train_detector_frcnn.py
│   ├── detect_damage.py
│   ├── generate_repair_plan.py
│   ├── render_visual_guidance.py
│   ├── capture_image.py
│   ├── repair_executor.py
│   ├── evaluate_model.py
│   ├── data_augmentation.py
│   ├── real_data_collector.py
│   └── model_optimization.py
│
├── utils/
│   ├── assembly_plan_utils.py
│   ├── visualization_utils.py
│   └── openai_utils.py
│
├── configs/
│   └── model_config.yaml
│
├── docs/
│   └── training_curve.png
│
├── requirements.txt
├── .env
├── .gitignore
├── main.py
└── README.md

```

---

## ⚙️ How It Works

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

# 1. Place your test image
Example:
```
./data/user_images/my_broken_chair.jpg
```

# 2. Run:
- You can run the pipeline either with a captured image from the Raspberry Pi camera or an uploaded image.

## 🔴 Option 1: Use Pi Camera
```bash
python main.py --camera
```

## 🔵 Option 2: Upload an Image
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```

---

## 🔍 What Happens Internally

1. Loads your image
2. Uses the trained model (`part_detector.pth`) to detect parts
3. Sends it to OpenAI GPT-4o via `analyze_image`
4. Outputs JSON file `./outputs/stage1_parts.json`


### The pipeline will:
- Detect broken parts using trained model (`Faster R-CNN` or `MobileNet`)
- Generate repair steps using `OpenAI GPT-4o`
- Output a repair graph for planning or robotics
- Render visual repair guides per part

---
### 📊 Training Curve
Here's the training progress (losses over 20 epochs): `./Repair2Skill/outputs/training_curves.png`


## 🧪 Other Tools & Utilities

- `train_detector_frcnn.py`: Train object detector using Faster R-CNN.
- `train_part_detector.py`: Train lightweight classifier model for parts/damages.
- `evaluate_model.py`: Evaluate model performance on test data.
- `generate_synthetic_data.py`: Generate labeled synthetic furniture damage dataset.
- `repair_executor.py`: Executes or simulates repair actions based on plan.
- `model_optimization.py`: Prepares models for low-power devices (e.g., quantization).
- `pose_estimation/estimate_pose.py`: Placeholder for pose estimation support.

---

## 🛠️ Setup Instructions

1. Clone and set up the environment
```bash
git clone <this_repo>
cd FurnitureRepairModel
python3 -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
---

## 📚 References
- Manual2Skill (RSS 2025): https://github.com/owensun2004/Manual2Skill
- PartNet Dataset (CVPR 2019): https://github.com/daerduoCarey/partnet_dataset
- IKEA-Manual Dataset (NeurIPS 2022): https://cs.stanford.edu/~kaichun/ikea.html
- Furniture-Assembly-Web Demo: https://owensun2004.github.io/Furniture-Assembly-Web/
