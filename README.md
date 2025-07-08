# 🛠️ Repair2Skill on Raspberry Pi 5

**Repair2Skill** is an AI-powered framework that enables robots (or intelligent agents) to detect and plan repairs for broken furniture parts. Inspired by [Manual2Skill (RSS 2025)](https://github.com/owensun2004/Manual2Skill), this project extends the original idea to handle **furniture repair** rather than just assembly—by using a camera, part detection, and **vision-language models (VLMs)** like GPT-4o.

This implementation runs on a **Raspberry Pi 5** equipped with a camera for real-time image capture.

---

## 🧠 What It Does

- Captures an image of a **broken chair** using a Pi Camera or uploaded image.
- Uses a trained **CNN (Faster R-CNN)** model to detect missing or broken parts.
- Sends the detection result to **GPT-4o** via OpenAI API to generate a **step-by-step repair plan**.
- Generates a **visual repair graph** to aid in execution or robot planning.

---

## 📁 Project Structure
```bash
FurnitureRepairModel/
│
├── data/
│   ├── partnet_data/
│   ├── ikea_manuals/
│   ├── synthetic_damage/
│   └── user_images/
│
├── models/
│   ├── damage_detection/
│   │   └── model.pth
│   ├── pose_estimation/
│   └── openai_integration/
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_part_detector.py
│   ├── detect_damage.py
│   ├── generate_repair_plan.py
│   ├── render_visual_guidance.py
│   └── capture_image.py
│
├── utils/
│   ├── assembly_plan_utils.py
│   ├── visualization_utils.py
│   └── openai_utils.py
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

## 🔵 Option 2: Upload an Image
```bash
python main.py --upload ./data/user_images/my_broken_chair.jpg
```
### The pipeline will:
  Detect broken parts
  Generate repair steps
  Output a repair graph (saved in outputs/)

## 🛠️ Setup Instructions
1. Clone and set up the environment
```bash
git clone 
cd 

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

``` 
2. Add your OpenAI API Key
Create a .env file in the root folder:
OPENAI_API_KEY=your-api-key-here



## 📚 References
- Manual2Skill (RSS 2025) Paper | GitHub https://github.com/owensun2004/Manual2Skill
- PartNet Dataset (CVPR 2019) GitHub https://github.com/daerduoCarey/partnet_dataset
- IKEA-Manual Dataset (NeurIPS 2022) Stanford Page https://cs.stanford.edu/~kaichun/ikea.html
- Product Disassembly Planning
  Goenka, P. et al. – for strategies relevant to disassembly & repair.
- Furniture-Assembly-Web Demo  | https://owensun2004.github.io/Furniture-Assembly-Web/
