# 🛠️ Repair2Skill — AI-Powered Robotic Furniture Repair

**Repair2Skill** is an AI-driven system that detects damaged furniture parts, generates structured repair plans, and simulates robotic repair execution in **PyBullet**.  
Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this framework extends the original concept from *assembly* to *repair*, combining **MobileNet SSD** detection, **GPT-4o** reasoning, and robotic simulation.

---

## 🧠 Core Capabilities

- Detects **damaged parts** in real furniture using **SSDLite-MobileNetV3**.  
- Generates **step-by-step repair plans** via GPT-4o.  
- Builds a **repair dependency graph** (hierarchy-aware).  
- Renders **visual repair guides** highlighting damaged and dependent parts.  
- Simulates the repair process using a robotic arm in **PyBullet** with **real-time streaming** and **interactive camera controls**.
- **Web Dashboard** with React frontend and Express.js backend for seamless pipeline execution.

---

## ⚙️ System Pipeline
```bash
Capture / Upload → Mobilenet SSD Detection → GPT-4o Repair Plan
                        ↓
Repair Graph Generation → Visual Guide → PyBullet Simulation (Auto-Start)
                        ↓
                    Real-Time Streaming with Interactive Camera Controls
```

| Stage  | Module                      | Output                                       |
|--------|-----------------------------|----------------------------------------------|
| 1      | `detect_damage.py`          | `stage1_parts.json`                          |
| 2      | `openai_utils.py`           | `repair_plan_<part>_<damage>.json`           |
| 3      | `repair_graph.py`           | `repair_graph_<part>.json` + `.png`          |
| 4      | `render_visual_guidance.py` | `data/visual_guides/<part>_repair_guide.png` |
| 5      | `run_simulation.py`         | Real-time PyBullet simulation streaming      |

---

## 📁 Project Structure
```bash
REPAIR2SKILL/
│
├── configs/
│   └── model_config.yaml
│
├── data/
│   ├── synthetic_damage/            # Generated training data
│   ├── user_images/                 # User-uploaded images
│   └── visual_guides/               # Generated visual repair guides
│
├── models/
│   └── damage_detection/
│       └── mobilenet_ssd.pth        # Trained model weights
│
├── outputs/                         # JSON plans, graphs, and detection logs
│   ├── stage1_parts.json
│   ├── repair_plan_*.json
│   └── repair_graph_*.json
│
├── pybullet_sim/                    # Simulation Engine
│   ├── __init__.py
│   ├── run_simulation.py            # Main simulation runner
│   ├── sim_connection.py            # PyBullet connection & camera
│   ├── sim_plan_executor.py         # Repair step execution
│   ├── sim_robot.py                 # Robot loading & IK
│   └── sim_scene.py                 # Chair scene construction
│
├── scripts/
│   ├── capture_image.py             # Camera capture utility
│   ├── chair_graph.py               # Chair dependency graph
│   ├── detect_damage.py             # MobileNet SSD inference
│   ├── generate_repair_plan.py      # GPT-4o integration
│   ├── generate_synthetic_data.py   # Synthetic data generation
│   ├── render_visual_guidance.py    # Visual guide rendering
│   ├── repair_graph.py              # Repair dependency graph
│   └── train_detector_mobilenet.py  # Model training script
│
├── utils/
│   └── openai_utils.py              # OpenAI API utilities
│
├── ui/                               # Web Interface (React + Express.js)
│   ├── src/
│   │   ├── App.js                   # React frontend
│   │   └── App.css                  # Styling
│   ├── public/
│   │   └── index.html
│   ├── server.js                    # Express.js backend
│   ├── package.json
│   └── uploads/                     # Uploaded images & plans
│
├── .env                              # Environment variables (API keys)
├── main.py                           # CLI pipeline entry point
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🔧 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Node.js 16+** (for web UI)
- **CUDA-capable GPU** (recommended for training)
- **OpenAI API key** for GPT-4o

### Step 1: Clone and set up Python environment
```bash
git clone <your_repo_url>
cd Repair2Skill
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Step 2: Set up Node.js dependencies for Web UI
```bash
cd ui
npm install
cd ..
```

### Step 3: Create a .env file in the root directory
```bash
OPENAI_API_KEY=sk-your-key-here
```

---

## 🚀 Usage

### Option A: Web Dashboard (Recommended)

The web dashboard provides an integrated experience with automatic pipeline execution and real-time simulation streaming.

#### Start the Web UI
```bash
cd ui
npm start
```

This will start:
- **React frontend** on `http://localhost:3000`
- **Express.js backend** on `http://localhost:3002`

#### Web UI Features
1. **Upload Image** → Select a damaged chair image
2. **Automatic Processing** → System automatically:
   - Detects damaged parts
   - Generates repair plan
   - Creates visual guide
   - **Auto-starts PyBullet simulation**
3. **Real-Time Streaming** → Watch the robot execute repairs live
4. **Interactive Camera Controls**:
   - **Zoom (Distance)**: Adjust camera distance (0.5 - 5.0)
   - **Rotate (Yaw)**: Rotate camera around scene (0° - 360°)
   - **Angle (Pitch)**: Change viewing angle (-90° - 90°)
   - Updates in real-time as you adjust sliders

### Option B: Command Line Interface

#### Step 1: Generate Synthetic Training Data
```bash
python main.py --generate-data --samples 1000
```

**Options:**
- `--samples N` — Number of synthetic chair images to generate (default: 1000)

**Output:**
- `data/synthetic_damage/images/` — Synthetic chair images
- `data/synthetic_damage/annotations.json` — Bounding boxes and damage labels

#### Step 2: Train the MobileNet SSD Detector Model
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

#### Step 3: Run the Full Pipeline (Detection + Repair Planning)

You can use either a **camera capture** or **upload an image** of a damaged chair.

##### Option A – Capture from Camera
```bash
python main.py --camera
```

##### Option B – Upload Existing Image
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg --threshold 0.25
```

**Options:**
- `--threshold FLOAT` — Confidence threshold for detections (default: 0.25, range: 0.0–1.0)

##### Pipeline Output
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

#### Step 4: Simulate the Repair in PyBullet

After running the pipeline, execute the robotic repair simulation:

```bash
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_back_left_leg_broken.json --robot kuka
```

**Options:**
- `--plan PATH` (required) — Path to the repair plan JSON
- `--graph PATH` (optional) — Path to repair graph JSON
- `--robot {kuka,panda}` — Robot arm type (default: kuka)
- `--damaged-part NAME` — Part to highlight in red (default: back_left_leg)
- `--camera-dist FLOAT` — Camera distance (default: 1.8)
- `--camera-yaw FLOAT` — Camera yaw angle (default: 40)
- `--camera-pitch FLOAT` — Camera pitch angle (default: -35)
- `--gui` — Enable PyBullet GUI window (default: headless mode)
- `--screenshot PATH` — Path to save periodic screenshots for streaming

##### Example Commands
```bash
# Run with KUKA robot (GUI enabled)
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_back_leg_broken.json --gui

# Run with Panda robot (headless with screenshot streaming)
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_armrest_right_broken.json --robot panda --screenshot ui/uploads/simulation_latest.jpg

# Run with custom camera settings
python pybullet_sim/run_simulation.py --plan outputs/repair_plan_seat_loose.json --camera-dist 2.5 --camera-yaw 60 --camera-pitch -20
```

##### Simulation Features
The simulation:
- **Loads a robotic arm** (KUKA iiwa or Panda)
- **Builds a simple chair** using colored block primitives
- **Highlights damaged parts** in red
- **Executes each repair step** in dependency order
- **Runs continuously** after plan completion for exploration
- **Real-time streaming** (6-7 FPS) when using screenshot mode
- **Visualizes actions:**
  - `remove/detach` → robot lifts part with gripper
  - `replace/attach` → robot installs new/repaired part
  - `tighten/screw` → robot wrist wiggles to simulate torque
  - `inspect` → brief pause for inspection

---

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
│  Stage 5: Robotic Simulation (Auto-Start)                                    │
│  ├─ PyBullet Physics Engine                                                  │
│  ├─ KUKA iiwa / Panda Arm                                                    │
│  ├─ Real-time repair execution                                               │
│  ├─ Continuous streaming (6-7 FPS)                                          │
│  └─ Interactive camera controls                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Key Module Structure

| Module | Purpose |
|--------|---------|
| `scripts/detect_damage.py` | MobileNet SSD inference for part/damage detection |
| `scripts/generate_repair_plan.py` | GPT-4o integration for repair planning |
| `scripts/repair_graph.py` | Build dependency graphs and visualizations |
| `scripts/render_visual_guidance.py` | Generate visual repair step guides |
| `scripts/train_detector_mobilenet.py` | Train MobileNet SSD detector model |
| `pybullet_sim/sim_connection.py` | PyBullet setup (physics, camera, lighting) |
| `pybullet_sim/sim_robot.py` | Robot loading and inverse kinematics |
| `pybullet_sim/sim_scene.py` | Chair construction and part highlighting |
| `pybullet_sim/sim_plan_executor.py` | Execute repair steps in order |
| `pybullet_sim/run_simulation.py` | Main simulation runner with streaming support |
| `ui/src/App.js` | React frontend for web dashboard |
| `ui/server.js` | Express.js backend API server |

### Example Pipeline Output
```
outputs/
├── stage1_parts.json                           # Detection results
├── repair_plan_back_left_leg_broken.json       # GPT-generated repair steps
├── repair_graph_back_left_leg.json             # Dependency graph
└── repair_graph_back_left_leg.png              # Graph visualization

data/visual_guides/
└── back_left_leg_repair_guide.png             # Step-by-step visual guide

ui/uploads/
├── repair_plan_*.json                          # Plan files for simulation
└── simulation_latest.jpg                       # Real-time simulation stream
```

---

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
  - Headless mode for web streaming

- **Procedural Scene** — `sim_scene.py` procedurally generates the chair geometry in PyBullet to ensure robust physics interactions, regardless of the input image.

- **Real-Time Streaming** — The simulation captures screenshots at 6-7 FPS and streams them to the web UI, enabling real-time visualization without opening a separate PyBullet window.

- **Interactive Camera Controls** — Users can adjust camera distance, yaw, and pitch in real-time through the web interface, with updates reflected immediately in the simulation stream.

- **Auto-Start Simulation** — After image upload and plan generation, the simulation automatically starts in the background, providing a seamless end-to-end experience.

---

## 🌐 Web Dashboard Architecture

The web dashboard uses a **React + Express.js** architecture:

- **Frontend (React)**: 
  - Interactive UI with pipeline visualization
  - Real-time image polling for simulation stream
  - Camera control sliders with live updates
  - Responsive design with modern CSS

- **Backend (Express.js)**:
  - RESTful API endpoints for image upload
  - Python subprocess management for pipeline execution
  - File serving for images and visual guides
  - Camera parameter management

- **Real-Time Streaming**:
  - PyBullet runs in headless mode
  - Screenshots saved every 150ms (~6-7 FPS)
  - UI polls for latest screenshot every 200ms
  - Camera parameters updated via JSON file

---

## 📚 References & Inspiration

- **Manual2Skill (RSS 2025)** — Original framework for robotic assembly, extended here to repair
- **PartNet Dataset (CVPR 2019)** — 3D part segmentation and structure
- **IKEA-Manual Dataset (NeurIPS 2022)** — Multi-step assembly instructions
- **PyBullet Simulator** — Open-source physics engine for robotics

---

## 🔮 Future Extensions

- [ ] Integrate **6D pose estimation** from point clouds for precise part localization
- [ ] Add **force feedback** and compliance control for realistic grasping
- [ ] Support **real 3D URDF models** (PartNet, IKEA dataset) instead of primitive shapes
- [ ] Implement **constraint-based repair** (welding, gluing, mechanical re-assembly)
- [ ] Deploy on **real robotic platforms** (Panda, KUKA, UR5)
- [ ] Add **reinforcement learning** to optimize repair strategies
- [ ] Support **multi-robot coordination** for complex repairs
- [ ] Enhanced camera controls with **preset views** and **smooth transitions**
- [ ] **Recording and playback** of repair simulations
- [ ] **Multi-view camera** support for better visualization

---

## 📄 License

This project is licensed under the MIT License — see LICENSE file for details.
