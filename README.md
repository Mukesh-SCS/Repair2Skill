# Repair2Skill on Raspberry Pi 5

Repair2Skill/
├── images/
├── camera/
├── gpt/
├── repair_graph/
├── robot_control/
└── README.md

## 🧠 Overview
This is a Raspberry Pi 5 project for detecting damaged chair parts using a camera and generating a visual repair plan using GPT-4o.

## 📁 Files
- `capture_image.py` – Captures image of the chair using the Pi camera
- `detect_parts.py` – Sends the image to OpenAI GPT-4o for part detection
- `generate_repair_graph.py` – Generates and displays a repair plan graph

## 🛠️ Setup
1. **Install system packages:**
```bash
sudo apt update
sudo apt install python3-picamera2 python3-pil -y
