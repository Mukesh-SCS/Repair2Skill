import os
import sys
import json
import logging
import cv2
import numpy as np
import time
import subprocess
import threading
import queue
from flask import Flask, render_template, request, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename

# -------------------------------------------------------------------
# 1. SETUP PATHS
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

SIM_DIR = os.path.join(BASE_DIR, "pybullet_sim")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

# -------------------------------------------------------------------
# 2. FLASK APP & GLOBALS
# -------------------------------------------------------------------
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "ui", "static"),
            template_folder=os.path.join(BASE_DIR, "ui", "templates"))

# Global State
LATEST_PLAN_PATH = None
LATEST_DAMAGED_PART = "seat"

# Camera State (Default View)
CAMERA_STATE = {
    "distance": 1.0,
    "yaw": 50,
    "pitch": -25,
    "target": [0.5, 0, 0.5]
}

# Global Lock to prevent multiple simulations from running at once
SIM_LOCK = threading.Lock()

# -------------------------------------------------------------------
# 3. ROUTES (MAIN & UPLOAD)
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", plan=None, guide=None, damaged_part=None)

@app.route("/upload", methods=["POST"])
def upload():
    global LATEST_PLAN_PATH, LATEST_DAMAGED_PART
    try:
        file = request.files["image"]
        if not file or file.filename == '':
            return "<h2>Error: No file selected</h2>", 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(BASE_DIR, "data", "user_images", filename)
        file.save(filepath)
        
        # Run Pipeline
        MAIN_PATH = os.path.join(BASE_DIR, "main.py")
        result = subprocess.run(
            [sys.executable, MAIN_PATH, "--upload", filepath],
            capture_output=True, text=True, timeout=300, cwd=BASE_DIR
        )

        if result.returncode != 0:
            return f"<h2>Pipeline Error</h2><pre>{result.stdout}\n{result.stderr}</pre>", 500

        # Load Results
        OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
        stage1 = json.load(open(os.path.join(OUTPUT_DIR, "stage1_parts.json")))
        pairs = stage1.get("detected_pairs", [])
        
        if not pairs:
            return render_template("index.html", plan="No damage detected.", guide=None)

        top = max(pairs, key=lambda x: x["damage_confidence"])
        part = top["part"]
        dmg_type = top["damage_type"]
        
        plan_path = os.path.join(OUTPUT_DIR, f"repair_plan_{part}_{dmg_type}.json")
        guide_path = f"data/visual_guides/{part}_repair_guide.png"

        LATEST_PLAN_PATH = plan_path
        LATEST_DAMAGED_PART = part

        return render_template("index.html",
                               plan=json.dumps(json.load(open(plan_path)), indent=2),
                               guide=guide_path,
                               damaged_part=part,
                               sim_ready=True) # Enables the sim view
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>", 500

# -------------------------------------------------------------------
# 4. CAMERA CONTROL API
# -------------------------------------------------------------------
@app.route("/update_camera")
def update_camera():
    """API to update camera angles from the UI sliders."""
    try:
        dist = request.args.get("dist", type=float)
        yaw = request.args.get("yaw", type=float)
        pitch = request.args.get("pitch", type=float)
        
        if dist is not None: CAMERA_STATE["distance"] = dist
        if yaw is not None: CAMERA_STATE["yaw"] = yaw
        if pitch is not None: CAMERA_STATE["pitch"] = pitch
        
        return jsonify({"status": "ok", "state": CAMERA_STATE})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------------------------------------------------
# 5. ROBUST SIMULATION WORKER
# -------------------------------------------------------------------
def simulation_worker(plan_path, damaged_part, frame_queue):
    with SIM_LOCK: # Ensure only one sim runs at a time
        try:
            import pybullet as p
            import pybullet_data
            import pybullet_sim.sim_connection as sim_conn
            import pybullet_sim.sim_robot as sim_robot
            import pybullet_sim.sim_plan_executor as sim_exec
            import pybullet_sim.sim_scene as sim_scene

            # --- DYNAMIC CAMERA CAPTURE FUNCTION ---
            def capture_frame():
                # Read latest global camera state
                vm = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=CAMERA_STATE["target"],
                    distance=CAMERA_STATE["distance"],
                    yaw=CAMERA_STATE["yaw"],
                    pitch=CAMERA_STATE["pitch"],
                    roll=0,
                    upAxisIndex=2
                )
                pm = p.computeProjectionMatrixFOV(60, 640/480, 0.1, 10.0)
                
                w, h, rgba, _, _ = p.getCameraImage(640, 480, vm, pm, renderer=p.ER_TINY_RENDERER)
                rgba = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
                frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                return None

            # --- MONKEY PATCH ---
            # Replaces step_sim to capture frames AND respect camera changes
            def smart_step_sim(seconds=0.4, hz=240):
                steps = int(seconds * hz)
                capture_every = 8 # 30 FPS
                
                for i in range(steps):
                    p.stepSimulation()
                    if i % capture_every == 0:
                        frame_data = capture_frame()
                        if frame_data:
                            try:
                                # Timeout=1s ensures we don't hang if browser disconnects
                                frame_queue.put(frame_data, timeout=1.0)
                            except queue.Full:
                                print("[SIM] Queue full, stopping thread.")
                                raise InterruptedError("Client Disconnected")

            # Apply Patch
            sim_conn.step_sim = smart_step_sim
            sim_robot.step_sim = smart_step_sim
            sim_exec.step_sim = smart_step_sim

            # --- SETUP SIM ---
            if p.isConnected(): p.disconnect()
            p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")

            # Load Scene
            robot_id, ee_idx, gripper_idx, open_val, close_val = sim_robot.load_robot("kuka")
            parts_dict = sim_scene.spawn_simple_chair(damaged_part)

            # Execute
            try:
                plan_data = json.load(open(plan_path))
                sequence = plan_data.get("repair_sequence", [])
            except:
                sequence = []

            # Loop indefinitely so user can zoom/rotate even after plan finishes
            # First, execute the plan
            smart_step_sim(1.0) # Settle
            for step in sequence:
                target = step.get("target_part", "")
                if target in parts_dict or target == "":
                    sim_exec.execute_step(robot_id, ee_idx, gripper_idx, open_val, close_val, parts_dict, step)
                    smart_step_sim(0.5)

            # Then, hold the final pose forever (for camera interaction)
            while True:
                smart_step_sim(0.1) 

        except InterruptedError:
            pass # Clean exit on disconnect
        except Exception as e:
            print(f"[SIM ERROR] {e}")
        finally:
            if 'p' in locals() and p.isConnected(): p.disconnect()
            frame_queue.put(None) # Signal end

@app.route("/simulation_feed")
def simulation_feed():
    if not LATEST_PLAN_PATH: return "No plan loaded", 404
    
    frame_queue = queue.Queue(maxsize=10)
    t = threading.Thread(target=simulation_worker, args=(LATEST_PLAN_PATH, LATEST_DAMAGED_PART, frame_queue))
    t.daemon = True
    t.start()

    def generator():
        while True:
            frame = frame_queue.get()
            if frame is None: break
            yield frame

    return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------------------------------------------
# 6. ASSET SERVING
# -------------------------------------------------------------------
@app.route("/static/<path:path>")
def send_static(path): return send_from_directory(os.path.join(BASE_DIR, "ui", "static"), path)

@app.route("/data/<path:path>")
def send_data(path): return send_from_directory(os.path.join(BASE_DIR, "data"), path)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='127.0.0.1')