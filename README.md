# 🛠️ Repair2Skill — AI-Powered Robotic Furniture Repair

**Repair2Skill** is an AI-driven system that detects damaged furniture parts, generates structured repair plans, and simulates robotic repair execution in **PyBullet**.  
Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this framework extends the original concept from *assembly* to *repair*, combining **MobileNet SSD** detection, **GPT-4o** reasoning, and robotic simulation.

---

## 🧠 Core Capabilities

- Detects **damaged parts** in real furniture using **SSDLite-MobileNetV3**.  
- Generates **step-by-step repair plans** via GPT-4o.  
- Builds a **repair dependency graph** (hierarchy-aware).  
- Renders **visual repair guides** highlighting damaged and dependent parts.  
- Simulates the repair process using a robotic arm in **PyBullet**, available via **CLI** or **Web Dashboard**.

---

## ⚙️ System Pipeline
```bash
Capture / Upload → Mobilenet SSD Detection → GPT-4o Repair Plan
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
│   ├── synthetic_damage/            #Generate training data
│   └── user_images/                 
│
├── models/
│   └── damage_detection/
│       └── mobilenet_ssd.pth        #Trained weights
│
├── outputs/                        #JSON plan , graphs and detection logs
│   ├── image.txt
│   └── stage1_parts.json
│
├── pybullet_sim/                  #Simulation Engine
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
├── ui/                       # Web Interface
│   ├── app.py                # Flask Backend (with Streaming)
│   ├── templates/            # HTML with Camera Controls
│   └── static/
├── .env
├── .gitignore
├── main.py                # Main Pipeline
├── QuickTest.py
├── QuickTest1.py
├── README.md
└── requirements.txt

```
---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- OpenAI API key for GPT-4o

### Step 1: Clone and set up the environment
```bash
git clone <your_repo_url>
cd Repair2Skill
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Step 2: Create a .env file in the root directory
```bash
OPENAI_API_KEY=sk-your-key-here
```
--- 

## 🚀 Usage

### Step 1: Generate Synthetic Training Data
```bash
python main.py --generate-data --samples 1000
```
**Options:**
- `--samples N` — Number of synthetic chair images to generate (default: 1000)

**Output:**
- `data/synthetic_damage/images/` — Synthetic chair images
- `data/synthetic_damage/annotations.json` — Bounding boxes and damage labels

### Step 2: Train the MobileNet SSD Detector Model
```bash
python main.py --train-frcnn --epochs 20 --batch 8
```

**Options:**
- `--epochs N` — Number of training epochs (default: 20)
- `--batch N` — Batch size (default: 2, adjust based on GPU memory)

**Model Architecture:**
- **Base Model:** SSDLite with MobileNetV3-Large backbone
- **Classes Detected:** 8 chair parts + 5 damage types = 13 classes
  - **Parts:** seat, back, front_left_leg, front_right_leg, back_left_leg, back_right_leg, armrest_left, armrest_right
  - **Damages:** missing, cracked, broken, loose, scratched

**Output:**
- `models/damage_detection/mobilenet_ssd.pth` — Trained model weights

### Step 3: Run the Full Pipeline (Detection + Repair Planning)

You can use either a **camera capture** or **upload an image** of a damaged chair.

#### Option A – Capture from Camera
```bash
python main.py --camera
```

#### Option B – Upload Existing Image
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg --threshold 0.25
```

**Options:**
- `--threshold FLOAT` — Confidence threshold for detections (default: 0.25, range: 0.0–1.0)

#### Pipeline Output
The pipeline generates:

1. **Detection Results** (`outputs/stage1_parts.json`)
   - Detected chair parts and damage locations
   - Confidence scores for each detection
   - Bounding box coordinates

2. **Repair Plan** (`outputs/repair_plan_<part>_<damage>.json`)
   - Step-by-step repair sequence generated by GPT-4o
   - Required tools, safety notes, and difficulty level
   - Dependency ordering (which parts to fix first)

3. **Repair Graph** (`outputs/repair_graph_<part>.json` + `.png`)
   - Visual dependency graph showing repair order
   - PNG visualization of part relationships

4. **Visual Repair Guide** (`data/visual_guides/<part>_repair_guide.png`)
   - Highlighted damaged and dependent parts
   - Step-by-step visual instructions

### Step 4: Simulate the Repair in PyBullet

After running the pipeline, execute the robotic repair simulation:

```bash
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_back_left_leg_broken.json --robot kuka
```

**Options:**
- `--plan PATH` (required) — Path to the repair plan JSON
- `--graph PATH` (optional) — Path to repair graph JSON
- `--robot {kuka,panda}` — Robot arm type (default: kuka)
- `--damaged-part NAME` — Part to highlight in red (default: back_left_leg)

#### Example Commands
```bash
# Run with KUKA robot (GUI enabled)
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_back_leg_broken.json

# Run with Panda robot
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_armrest_right_broken.json --robot panda
```

#### Simulation Features
The simulation:
- **Loads a robotic arm** (KUKA iiwa or Panda)
- **Builds a simple chair** using colored block primitives
- **Highlights damaged parts** in red
- **Executes each repair step** in dependency order
- **Visualizes actions:**
  - `remove/detach` → robot lifts part (colors it orange)
  - `replace/attach` → recolors part green
  - `tighten/screw` → robot wrist wiggles to simulate torque
  - `inspect` → brief pause for inspection

### Step 5: Web UI Dashboard 

Alternatively, run the interactive Flask web interface for an easier workflow:

```bash
python ui/app.py
```

Then open **http://localhost:5000** in your browser.

#### Features
- **Upload chair images** via drag-and-drop
- **View detection results** with confidence scores
- **Generate repair plans** with one click
- **Inspect repair graphs** visually
- **Download results** as JSON files



## 📊 Full System Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         REPAIR2SKILL PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: Image Input          │  Stage 2: Detection                         │
│  ├─ Camera Capture             │  ├─ Load MobileNet SSD model                │
│  └─ File Upload                │  └─ outputs/stage1_parts.json               │
│                                │                                             │
│  Stage 3: Repair Planning      │  Stage 4: Graph & Visualization             │
│  ├─ GPT-4o LLM                 │  ├─ Build dependency graph                  │
│  └─ outputs/repair_plan_*.json │  ├─ Render visual guides                    │
│                                │  └─ outputs/repair_graph_*.json             │
│                                │                                             │
│  Stage 5: Robotic Simulation                                                 │
│  ├─ PyBullet Physics Engine                                                  │
│  ├─ KUKA iiwa / Panda Arm                                                    │
│  └─ Real-time repair execution                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Key Module Structure

| Module | Purpose |
|--------|---------|
| `scripts/detect_damage.py` | MobileNet SSD inference for part/damage detection |
| `scripts/generate_repair_plan.py` | GPT-4o integration for repair planning |
| `scripts/repair_graph.py` | Build dependency graphs and visualizations |
| `scripts/render_visual_guidance.py` | Generate visual repair step guides |
| `pybullet_sim/sim_connection.py` | PyBullet setup (physics, camera, lighting) |
| `pybullet_sim/sim_robot.py` | Robot loading and inverse kinematics |
| `pybullet_sim/sim_scene.py` | Chair construction and part highlighting |
| `pybullet_sim/sim_plan_executor.py` | Execute repair steps in order |
| `ui/app.py` | Flask web interface |

### Example Pipeline Output
```
outputs/
├── stage1_parts.json                           # Detection results
├── repair_plan_back_left_leg_broken.json       # GPT-generated repair steps
├── repair_graph_back_left_leg.json             # Dependency graph
├── repair_graph_back_left_leg.png              # Graph visualization
data/visual_guides/
└── back_left_leg_repair_guide.png             # Step-by-step visual guide
```


## 🔑 Key Concepts

- **MobileNet SSD Detector** — Fast, lightweight object detection optimized for furniture parts and specific damage types.
  
- **Repair Dependency Graph** — Logic layer that enforces physical constraints (e.g., "Must remove seat before fixing leg").
  
- **GPT-4o Repair Planning** — Generates structured, step-by-step repair instructions with:
  - Required tools and materials
  - Safety warnings
  - Difficulty levels
  - Time estimates
  
- **PyBullet Physics Simulation** — Visualizes robotic repair execution with:
  - Real-time gripper movements
  - Part manipulation and reattachment
  - Torque simulation (for tightening/screwing)
  - Collision detection

- **Procedural Scene**  — sim_scene.py procedurally generates the chair geometry in PyBullet to ensure robust physics interactions, regardless of the input image.

- **Streaming Engine**  — app.py uses a threaded generator to capture PyBullet frames and stream them via MJPEG, enabling "headless" simulation rendering in the browser.


## 📚 References & Inspiration

- **Manual2Skill (RSS 2025)** — Original framework for robotic assembly, extended here to repair
- **PartNet Dataset (CVPR 2019)** — 3D part segmentation and structure
- **IKEA-Manual Dataset (NeurIPS 2022)** — Multi-step assembly instructions
- **PyBullet Simulator** — Open-source physics engine for robotics

## 🔮 Future Extensions

- [ ] Integrate **6D pose estimation** from point clouds for precise part localization
- [ ] Add **force feedback** and compliance control for realistic grasping
- [ ] Support **real 3D URDF models** (PartNet, IKEA dataset) instead of primitive shapes
- [ ] Implement **constraint-based repair** (welding, gluing, mechanical re-assembly)
- [ ] Deploy on **real robotic platforms** (Panda, KUKA, UR5)
- [ ] Add **reinforcement learning** to optimize repair strategies
- [ ] Support **multi-robot coordination** for complex repairs

## 📄 License

This project is licensed under the MIT License — see LICENSE file for details.
