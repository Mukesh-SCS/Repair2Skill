# 🛠️ Repair2Skill — AI-Powered Robotic Furniture Repair

**Repair2Skill** is an AI-driven system that detects damaged furniture parts, generates structured repair plans, and simulates robotic repair execution in **PyBullet**.  
Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this framework extends the original concept from *assembly* to *repair*, combining **Faster R-CNN** detection, **GPT-4o** reasoning, and robotic simulation.

---

## 🧠 Core Capabilities

- Detects **damaged parts** in real furniture using Faster R-CNN.  
- Generates **step-by-step repair plans** via GPT-4o.  
- Builds a **repair dependency graph** (Manual2Skill-style hierarchy).  
- Renders **visual repair guides** highlighting damaged and dependent parts.  
- Simulates the repair process using a robotic arm in **PyBullet**.

---

## ⚙️ System Pipeline
```bash
Capture / Upload → FRCNN Detection → GPT-4o Repair Plan
                        ↓
Repair Graph Generation → Visual Guide → PyBullet Simulation

---------------------------------------------------------------------------------------
| Stage  | Module                      | Output                                       |
|--------|-----------------------------|----------------------------------------------|
| 1      | `detect_damage.py`          | `stage1_parts.json`                          |
| 2      | `openai_utils.py`           | `repair_plan_<part>_<damage>.json`           |
| 3      | `repair_graph.py`           | `repair_graph_<part>.json` + `.png`          |
| 4      | `render_visual_guidance.py` | `data/visual_guides/<part>_repair_guide.png` |
| 5      | `robot_executor.py`         |  Simulated robotic repair execution          |
---------------------------------------------------------------------------------------
```
---

## 📁 Project Structure
```bash
Repair2Skill/
├── data/
│ ├── synthetic_damage/
│ └── user_images/
│
├── models/
│ └── damage_detection/
│
├── outputs/
│ ├── stage1_parts.json
│ ├── repair_plan_<part><damage>.json
│ ├── repair_graph<part>.json
│ └── repair_graph_<part>.png
│
├── scripts/
│ ├── capture_image.py
│ ├── detect_damage.py
│ ├── train_detector_frcnn.py
│ ├── generate_synthetic_data.py
│ ├── generate_repair_plan.py
│ ├── render_visual_guidance.py
│ ├── repair_graph.py
│ └── robot_executor.py
│
├── utils/
│ └── openai_utils.py
│
├── main.py
├── requirements.txt
└── README.md
```
---

## 🔧 Installation & Setup

1. Clone and set up the environment:
```bash
   git clone <your_repo_url>
   cd FurnitureRepairModel
   python3 -m venv .venv
   source .venv/scripts/activate
   pip install -r requirements.txt

```
## Create a .env file in the root directory:
```bash
OPENAI_API_KEY=sk-...
```
--- 

## Usage
### Step 1: Generate Synthetic Training Data
```bash
python main.py --generate-data --samples 500
```
- Generates images/ and annotations.json under ./data/synthetic_damage/.

### Step 2: Train the Faster R-CNN Model
```bash
python main.py --train-frcnn
```

- Model saved to:
```bash
./models/damage_detection/frcnn_model.pth
``` 
---

## 🚀 Step 3: Run the Full Pipeline

You can use either a captured image (camera) or upload one.

### Option 1 – Camera
```bash
python main.py --camera
```
## Option 2 – Upload Image
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```
### This will:
- Detect damaged parts
- Generate a GPT-4o repair plan
- Create a hierarchical repair graph
- Render a visual repair guide


- Step 4: Simulate the Repair in PyBullet
- After running the pipeline:
```bash
python scripts/robot_executor.py
```
#### The simulation will:
- Load the robotic arm (KUKA iiwa)
- Parse your repair_plan_*.json and repair_graph_*.json
- Execute each repair step in dependency order

The simulation will:

Load the robotic arm (KUKA iiwa)

Parse your repair_plan_*.json and repair_graph_*.json

###  Execute each repair step in dependency order
```bash
outputs/
 ├── stage1_parts.json
 ├── repair_plan_back_leg_broken.json
 ├── repair_graph_back_leg.json
 ├── repair_graph_back_leg.png
data/visual_guides/
 ├── back_leg_repair_guide.png
```

## Key Concepts
- Repair Graph — Built using chair_graph.py and repair_graph.py; ensures repairs follow mechanical dependencies
(e.g., remove leg → fix → reattach).
- Faster R-CNN — Used for part + damage detection (higher accuracy than MobileNet).
- GPT-4o — Generates structured repair steps, required tools, and safety guidance.
- PyBullet — Executes the full plan using a simulated robotic arm.

## References
- Manual2Skill (RSS 2025)
- PartNet Dataset (CVPR 2019)
- IKEA-Manual Dataset (NeurIPS 2022)
- PyBullet Simulator

### Next Steps
- Integrate pose estimation from visual + point cloud data
- Extend PyBullet actions with grasping and force feedback
- Add dynamic 3D chair URDF models for realistic repair interaction