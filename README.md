# Repair2Skill on Raspberry Pi 5

**Repair2Skill** is an AI-powered framework that enables robots (or intelligent agents) to perform real-world furniture repair tasks. Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this project adapts the approach to repair broken furniture using a combination of images, manual texts, and vision-language models (VLMs) like GPT-4o.


## 🧠 Overview
This is a Raspberry Pi 5 project for detecting damaged chair parts using a camera and generating a visual repair plan using GPT-4o.


---
## 📦 Project Directory Structure

FurnitureRepairModel/
│
├── data/
│   ├── partnet_data/           # Downloaded from PartNet Dataset
│   ├── ikea_manuals/           # From IKEA manual dataset
│   └── synthetic_damage/       # Synthetic damaged scenarios
│
├── models/
│   ├── damage_detection/       # CNN-based part detector
│   ├── pose_estimation/        # Pose Estimator (from Manual2Skill)
│   └── openai_integration/     # API integration with GPT-4
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_part_detector.py
│   ├── detect_damage.py
│   ├── generate_repair_plan.py
│   └── render_visual_guidance.py
│
├── utils/
│   ├── assembly_plan_utils.py
│   ├── visualization_utils.py
│   └── openai_utils.py
│
├── requirements.txt
└── main.py                      # Entry point for the pipeline





## 🛠️ Setup
1. **Install system packages:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Link : https://owensun2004.github.io/Furniture-Assembly-Web/

```


``` Sources:
Tie et al. “Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly…” RSS 2025 – (for base framework and ideas on using VLMs for assembly)
arxiv.org
github.com
IKEA-Manual Dataset (Wang et al., NeurIPS 2022) – (for paired 3D furniture models and assembly manuals with annotations)
cs.stanford.edu
cs.stanford.edu
PartNet Dataset (Mo et al., CVPR 2019) – (for fine-grained part-level 3D models, used in training pose estimation and segmentation)
github.com
github.com
Manual2Skill GitHub Repository – (implementation details for assembly graph generation and pose estimation pipelines)
github.com
github.com
Piyush Goenka, “Product Disassembly Sequence Planning” – (discussion of AI-driven disassembly, relevant to planning removal steps in repairs) 