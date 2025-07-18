# 🛠️ Repair2Skill on Raspberry Pi 5

**Repair2Skill** is an AI-powered framework that enables robots (or intelligent agents) to detect and plan repairs for broken furniture parts. Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this project extends the original idea to handle **furniture repair** by using a camera, part detection, and offline rule-based repair planning.

This implementation runs on a **Raspberry Pi 5** equipped with a camera for real-time image capture.

---

## 🧠 What It Does

- Captures an image of a **broken chair** using a Pi Camera or uploaded image.
- Uses a trained **MobileNet-based classifier** to detect damaged parts.
- Generates **step-by-step repair plans locally** without external APIs.
- Produces **visual repair guides** for robotic or manual execution.
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
│   └── pose_estimation/
│
├── outputs/
│   ├── repair_plans/
│   ├── visual_guides/
│   └── detection_results/
│
├── scripts/
│   ├── data_generation/
│   │   └── generate_synthetic_data.py
│   ├── training/
│   │   ├── train_part_detector.py
│   │   └── train_detector_frcnn.py  # optional / archive
│   ├── detection/
│   │   └── detect_damage.py
│   ├── repair/
│   │   ├── generate_repair_plan.py
│   │   ├── repair_executor.py
│   │   └── render_visual_guidance.py
│   ├── utils/
│       ├── capture_image.py
│       ├── evaluate_model.py
│       ├── data_augmentation.py
│       ├── real_data_collector.py
│       └── model_optimization.py
│
├── utils/
│   ├── assembly_plan_utils.py
│   └── visualization_utils.py
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

### Step 1:  Generate Synthetic Data
```bash
python main.py --generate-data --samples 2000
```

Generates labeled synthetic images with various damaged chair parts, saved in ./data/synthetic_damage/.

### Step 2: Train the Model

```bash
python main.py --train
```
Trains the damage and part detection model on the synthetic data and saves weights to:

```bash
./models/damage_detection/part_detector.pth
```
---

### Step 3: Run the Detection and Repair Pipeline
Place your test image in ./data/user_images/ (e.g., my_broken_chair.jpg), then run:

```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```
Or capture a new image using Raspberry Pi camera:

```bash
python main.py --camera
```
--- 


## 🔍 What Happens Internally

1. Loads your image.
2. Detects damaged parts using the trained model.
3. Generates a detailed repair plan locally using static rules.
4. Produces visual guides highlighting parts to repair.
5. Saves outputs under ./outputs/ and ./data/visual_guides/.

---
### 📊 Training Curve
Here's the training progress (losses over 20 epochs): `./Repair2Skill/outputs/training_curves.png`


## 🧪 Additional Tools

- `train_detector_frcnn.py`: Faster R-CNN training (optional).
- `evaluate_model.py`: Evaluate model accuracy.
- `generate_synthetic_data.py`: Synthetic dataset creation.
- `repair_executor.py`: Simulates repair execution.
- `model_optimization.py`: Model quantization for low-power devices.



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


# Feel free to explore, train, and repair your broken chairs fully offline with Repair2Skill!