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
REPAIR2SKILL/
│
├── configs/
│   └── model_config.yaml
│
├── data/
│   ├── synthetic_damage/
│   └── user_images/
│
├── models/
│   └── damage_detection/
│       └── frcnn_model.pth
│
├── outputs/
│   ├── image.txt
│   └── stage1_parts.json
│
├── pybullet_sim/
│   ├── __init__.py
│   ├── run_simulation.py
│   ├── sim_connection.py
│   ├── sim_plan_executor.py
│   ├── sim_robot.py
│   ├── sim_scene.py
│
├── scripts/
│   ├── __pycache__/
│   ├── capture_image.py
│   ├── chair_graph.py
│   ├── detect_damage.py
│   ├── generate_repair_plan.py
│   ├── generate_synthetic_data.py
│   ├── render_visual_guidance.py
│   ├── repair_graph.py
│   └── train_detector_frcnn.py
│
├── utils/
│   ├── __pycache__/
│   └── openai_utils.py
│
├── .env
├── .gitignore
├── main.py
├── QuickTest.py
├── QuickTest1.py
├── README.md
└── requirements.txt

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
## **PyBullet Simulation (Stage 5)**
This module executes the Repair2Skill robotic repair simulation.
It visualizes how a robot would perform each step of the GPT-generated repair plan using PyBullet physics.

### How to Run
```bash 
example
python pybullet/run_simulation.py --plan outputs/repair_plan_back_left_leg_broken.json
```
### Optional arguments:
```bash
----------------------------------------------------------------
|Flag	      |                Description                       |
|-----------|--------------------------------------------------|
|--graph	   |      Path to repair graph JSON (optional)        |
|--robot    |    {kuka,panda}	Choose robot arm (default: kuka) |
|--headless	|     Run without GUI (for CI or remote)           |
----------------------------------------------------------------

Example 
python pybullet/run_simulation.py --robot panda --headless

```
## What Happens
- Connects to PyBullet and sets up gravity, lighting, and camera.
- Loads a robot arm (KUKA iiwa or Panda).
- Builds a simple chair using colored block primitives.
     - Red = damaged part
     - Gray = normal parts
- Reads your repair plan (repair_plan_*.json) and optional repair graph.

**Simulates each step:**
- remove / detach → robot lifts the part and recolors it orange.
- replace / attach → recolors the part green.
- tighten / screw → robot wrist wiggles to simulate torque.
- inspect → brief pause.
- Keeps the simulation open until you close the window.



## Modular Design
```bash
----------------------------------------------------------------------------------------
| File                   | Purpose                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `sim_connection.py`    | Core simulator setup (camera, physics, stepping)            |
| `sim_robot.py`         | Robot loading, gripper, and inverse kinematics              |
| `sim_scene.py`         | Scene construction (chair model, damaged part highlighting) |
| `sim_plan_executor.py` | Executes actions from the repair plan in dependency order   |
| `run_simulation.py`    | Integrates all modules and runs the repair process          |
----------------------------------------------------------------------------------------
```
**This structure makes it easy to:**
Debug or extend one part (e.g., replace robot with UR5).
Run simulation tests separately.
Add more complex repair environments later.

**Example Output**
When you run the simulation:
The robot arm appears beside a simple chair model.
Damaged parts are red.
As each repair step executes, the robot moves near the part, recolors it, and proceeds through the plan.


## Next Extensions
Replace block-based chair with a real URDF model (from PartNet or IKEA dataset).
Add constraint-based repair (re-attach parts physically).
Integrate with real grasping for the Panda gripper.
Support 6D pose input from your detection model.

## Purpose:
This stage demonstrates the execution phase of the Repair2Skill pipeline — turning AI-generated repair instructions into visible robotic actions.


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