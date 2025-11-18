import os
import subprocess
import json
import logging
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "user_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GUIDE_DIR = os.path.join(BASE_DIR, "data", "visual_guides")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GUIDE_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "ui", "static"),
            template_folder=os.path.join(BASE_DIR, "ui", "templates"))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           plan=None,
                           guide=None,
                           damaged_part=None)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["image"]
        if not file or file.filename == '':
            return "<h2>Error: No file selected</h2>", 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        
        # Run the pipeline
        result = subprocess.run(
            [os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"), MAIN_PATH, "--upload", filepath],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=BASE_DIR
        )

        if result.returncode != 0:
            return f"""
                <h2>Pipeline Error</h2>
                <h3>STDOUT:</h3>
                <pre>{result.stdout}</pre>
                <h3>STDERR:</h3>
                <pre>{result.stderr}</pre>
            """, 500

        # Load detection results
        stage1_path = os.path.join(OUTPUT_DIR, "stage1_parts.json")
        if not os.path.exists(stage1_path):
            return render_template("index.html",
                                   plan="Error: Detection results not found.",
                                   guide=None,
                                   damaged_part=None)
        
        stage1 = json.load(open(stage1_path))
        pairs = stage1.get("detected_pairs", [])

        if not pairs:
            return render_template("index.html",
                                   plan="No damaged parts detected.",
                                   guide=None,
                                   damaged_part=None)

        # Pick highest-confidence damaged part
        top = max(pairs, key=lambda x: x["damage_confidence"])
        part = top["part"]
        dmg_type = top["damage_type"]

        # Paths
        plan_path = os.path.join(OUTPUT_DIR, f"repair_plan_{part}_{dmg_type}.json")
        guide_filename = f"{part}_repair_guide.png"
        guide_path = f"data/visual_guides/{guide_filename}"

        if not os.path.exists(plan_path):
            return render_template("index.html",
                                   plan=f"Error: Repair plan not found at {plan_path}",
                                   guide=None,
                                   damaged_part=part)

        plan = json.load(open(plan_path))

        return render_template("index.html",
                               plan=json.dumps(plan, indent=2),
                               guide=guide_path,
                               damaged_part=part)
    
    except subprocess.TimeoutExpired:
        return "<h2>Error: Pipeline timed out (>5 minutes)</h2>", 500
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>", 500


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory(os.path.join(BASE_DIR, "ui", "static"), path)

@app.route("/data/<path:path>")
def send_data(path):
    return send_from_directory(os.path.join(BASE_DIR, "data"), path)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='127.0.0.1')
