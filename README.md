# 🛠️ Repair2Skill on Raspberry Pi 5

**Repair2Skill** is an AI-powered framework that enables robots (or intelligent agents) to detect and plan repairs for broken furniture parts. Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this project extends the original idea to handle **furniture repair** by using a camera, part detection, and **vision-language models (VLMs)** like GPT-4o.

This implementation runs on a **Raspberry Pi 5** equipped with a camera for real-time image capture.

---

## 🧠 What It Does

- Captures an image of a **broken chair** using a Pi Camera or uploaded image.
- Uses a trained **Faster R-CNN** model to detect damaged parts.
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
├── requirements.txt
├── .env
├── .gitignore
├── main.py
└── README.md
```

---

## ⚙️ How It Works

You can run the pipeline either with a captured image from the Raspberry Pi camera or an uploaded image.

### 🔴 Option 1: Use Pi Camera
```bash
python main.py --camera
```

### 🔵 Option 2: Upload an Image
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```

### The pipeline will:
- Detect broken parts using `Faster R-CNN`
- Generate repair steps using `OpenAI GPT-4`
- Output a repair graph for planning or robotics
- Render visual repair guides per part

---

## 🧪 Other Tools & Utilities

- `train_detector_frcnn.py`: Train damage detection using Faster R-CNN.
- `evaluate_model.py`: Evaluate trained models on test data.
- `data_augmentation.py`: Augment datasets using albumentations.
- `real_data_collector.py`: Collect & label real-world damaged furniture samples.
- `model_optimization.py`: Quantize/prune models for Raspberry Pi deployment.
- `repair_executor.py`: Simulate or perform repair steps based on generated graph.
- `pose_estimation/estimate_pose.py`: Placeholder for future pose estimation integration.

---

## 🛠️ Setup Instructions

1. Clone and set up the environment
```bash
git clone <this_repo>
cd FurnitureRepairModel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Add your OpenAI API Key in `.env`
```
OPENAI_API_KEY=your-api-key-here
```

---

## 📚 References
- Manual2Skill (RSS 2025): https://github.com/owensun2004/Manual2Skill
- PartNet Dataset (CVPR 2019): https://github.com/daerduoCarey/partnet_dataset
- IKEA-Manual Dataset (NeurIPS 2022): https://cs.stanford.edu/~kaichun/ikea.html
- Furniture-Assembly-Web Demo: https://owensun2004.github.io/Furniture-Assembly-Web/
