# Repair2Skill on Raspberry Pi 5

**Repair2Skill** is an AI-powered framework that enables robots (or intelligent agents) to perform real-world furniture repair tasks. Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this project adapts the approach to repair broken furniture using a combination of images, manual texts, and vision-language models (VLMs) like GPT-4o.


## 🧠 Overview
This is a Raspberry Pi 5 project for detecting damaged chair parts using a camera and generating a visual repair plan using GPT-4o.


---
## 📦 Project Structure

Repair2Skill/
├── data/
│ ├── images/ # Captured or test images of broken furniture
│ ├── annotations.json # Part-level condition (e.g., broken, intact)
│ └── repair_manuals/ # Text-based repair instructions
├── openai_prompts/
│ └── prompt_repair_graph.txt
├── repair_graph_gen/
│ └── repair_run.py # CLI tool to generate repair plans
├── web/ # Repair2Skill web interface (React + Tailwind)
│ └── components/
│ └── Repair2SkillApp.tsx
├── README.md



## 🛠️ Setup
1. **Install system packages:**
```bash
sudo apt update
sudo apt install python3-picamera2 python3-pil -y


Link : https://owensun2004.github.io/Furniture-Assembly-Web/









## 🔍 What It Does

1. 📸 Takes an image of a broken furniture item (e.g., chair)
2. 🤖 Uses GPT-4o to analyze broken parts (from annotations or detection)
3. 🧠 Generates a hierarchical **repair plan**
4. 🌐 Optionally visualizes the repair plan in a clean web UI

---


### ✅ Prerequisites

- Python 3.9+
- Node.js + npm
- OpenAI API key (GPT-4o)

---

### 🧪 CLI: Generate Repair Plan

```bash
cd repair_graph_gen
python3 repair_run.py
⚙️ This will load part annotations, query GPT-4o using your custom prompt, and print the repair steps.

💻 Web UI
The UI allows image upload and repair plan generation.

### Install & Run

cd web
npm install
npm run dev 

Then open http://localhost:3000 in your browser.