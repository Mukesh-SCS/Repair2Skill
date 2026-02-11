"""
run_simulation.py — Load repair plan JSON and run full simulation with Panda and chair.
Usage:
  python pybullet_sim/run_simulation.py --plan outputs/repair_plan_back_left_leg_broken.json --gui
  python pybullet_sim/run_simulation.py --plan ... --stream-port 8080 --camera-params path/to/camera_params.json
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import threading
import time

# Run from repo root so imports work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from pybullet_sim.sim_connection import connect, reset_camera, step_sim, get_client
    from pybullet_sim.sim_robot import load_robot, open_gripper
    from pybullet_sim.sim_scene import ChairScene
    from pybullet_sim.sim_plan_executor import execute_step
except ImportError:
    from sim_connection import connect, reset_camera, step_sim, get_client
    from sim_robot import load_robot, open_gripper
    from sim_scene import ChairScene
    from sim_plan_executor import execute_step

import pybullet as p

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SIM_HZ = 240.0
PANDA_BASE = (0.0, 0.0, 0.0)
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_INTERVAL = 0.15  # ~6–7 FPS

# Shared buffer for streamed frame (JPEG bytes)
_latest_frame_jpeg = None
_frame_lock = threading.Lock()


def load_plan(path):
    """Load repair_sequence from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seq = data.get("repair_sequence") or data.get("repair_plan") or []
    return seq


def _read_camera_params(path):
    """Read dist, yaw, pitch from JSON file. Returns None if file missing/invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {
            "dist": float(d.get("dist", 2.4)),
            "yaw": float(d.get("yaw", 55.0)),
            "pitch": float(d.get("pitch", -25.0)),
        }
    except Exception:
        return None


def _capture_frame(cid, dist, yaw, pitch, target=(0.35, 0.0, 0.35)):
    """Render current view to RGB and return as JPEG bytes (for headless streaming)."""
    try:
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        dx = dist * math.cos(pitch_rad) * math.sin(yaw_rad)
        dy = dist * math.cos(pitch_rad) * math.cos(yaw_rad)
        dz = dist * math.sin(pitch_rad)
        eye = (target[0] + dx, target[1] + dy, target[2] + dz)
        up = (0, 0, 1)
        view = p.computeViewMatrix(eye, target, up, physicsClientId=cid)
        fov = 60
        aspect = STREAM_WIDTH / float(STREAM_HEIGHT)
        near, far = 0.01, 10.0
        proj = p.computeProjectionMatrixFOV(fov, aspect, near, far, physicsClientId=cid)
        _, _, rgb_flat, _, _ = p.getCameraImage(
            STREAM_WIDTH, STREAM_HEIGHT, viewMatrix=view, projectionMatrix=proj, physicsClientId=cid
        )
        if rgb_flat is None:
            return None
        try:
            import numpy as np
            from PIL import Image
            rgba = np.array(rgb_flat, dtype=np.uint8).reshape((STREAM_HEIGHT, STREAM_WIDTH, 4))
            rgb = rgba[:, :, :3]
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception as e:
            logger.debug("Frame encode failed: %s", e)
            return None
    except Exception as e:
        logger.debug("Capture frame failed: %s", e)
        return None


def _run_stream_server(port):
    """Run a simple HTTP server that serves GET /frame.jpg with the latest JPEG."""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
    global _latest_frame_jpeg

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0].rstrip("/") == "/frame.jpg":
                with _frame_lock:
                    data = _latest_frame_jpeg
                if data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(204)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, format, *args):
            pass
    server = HTTPServer(("127.0.0.1", port), Handler)
    logger.info("Stream server listening on http://127.0.0.1:%s/frame.jpg", port)
    server.serve_forever()

    server.server_close()


def main():
    global _latest_frame_jpeg
    ap = argparse.ArgumentParser(description="Run PyBullet chair repair simulation")
    ap.add_argument("--plan", type=str, required=True, help="Path to repair plan JSON")
    ap.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    ap.add_argument("--damaged-part", type=str, default=None, help="Part to highlight (default: from first step)")
    ap.add_argument("--camera-dist", type=float, default=2.4, help="Camera distance")
    ap.add_argument("--camera-yaw", type=float, default=55.0)
    ap.add_argument("--camera-pitch", type=float, default=-25.0)
    ap.add_argument("--stream-port", type=int, default=None, help="Serve GET /frame.jpg on this port (headless)")
    ap.add_argument("--camera-params", type=str, default=None, help="JSON file to read camera dist/yaw/pitch from")
    ap.add_argument("--screenshot", type=str, default=None, help="Path to save periodic screenshot (optional)")
    args = ap.parse_args()

    plan_path = os.path.abspath(args.plan)
    if not os.path.isfile(plan_path):
        logger.error("Plan file not found: %s", plan_path)
        sys.exit(1)

    steps = load_plan(plan_path)
    if not steps:
        logger.error("No repair_sequence in plan")
        sys.exit(1)

    damaged_part = args.damaged_part
    if not damaged_part and steps:
        damaged_part = (steps[0].get("target_part") or "back_left_leg").strip()
    logger.info("Damaged part (highlight): %s", damaged_part)

    use_gui = args.gui and not args.stream_port
    connect(gui=use_gui)
    cid = get_client()
    reset_camera(
        dist=args.camera_dist,
        yaw=args.camera_yaw,
        pitch=args.camera_pitch,
        target=(0.35, 0.0, 0.35),
    )

    robot_id, ee_link, gripper_joints = load_robot("panda", base_position=PANDA_BASE)
    open_gripper(robot_id, gripper_joints, 0.04)
    step_sim(0.5, SIM_HZ, blocking=True)

    scene = ChairScene(chair_center=(0.6, 0.0, 0.0), damaged_part=damaged_part)
    step_sim(0.5, SIM_HZ, blocking=True)

    for step in steps:
        step_id = step.get("step_id", "?")
        action = step.get("action_type", "")
        target = step.get("target_part", "")
        logger.info("Executing step %s: %s %s", step_id, action, target)
        try:
            execute_step(robot_id, ee_link, gripper_joints, scene, step)
        except Exception as e:
            logger.exception("Step %s failed: %s", step_id, e)
    logger.info("Plan finished. Simulation continues.")
    step_sim(2.0, SIM_HZ, blocking=True)

    if args.stream_port:
        # Start HTTP server thread for /frame.jpg
        server_thread = threading.Thread(target=_run_stream_server, args=(args.stream_port,), daemon=True)
        server_thread.start()
        time.sleep(0.3)
        cam_dist, cam_yaw, cam_pitch = args.camera_dist, args.camera_yaw, args.camera_pitch
        target = (0.35, 0.0, 0.35)
        try:
            while True:
                params = _read_camera_params(args.camera_params)
                if params:
                    cam_dist, cam_yaw, cam_pitch = params["dist"], params["yaw"], params["pitch"]
                reset_camera(dist=cam_dist, yaw=cam_yaw, pitch=cam_pitch, target=target)
                step_sim(1.0 / 30.0, SIM_HZ, blocking=True)
                jpeg = _capture_frame(cid, cam_dist, cam_yaw, cam_pitch, target)
                if jpeg:
                    with _frame_lock:
                        _latest_frame_jpeg = jpeg
                if args.screenshot and jpeg and os.path.isdir(os.path.dirname(args.screenshot)):
                    try:
                        with open(args.screenshot, "wb") as f:
                            f.write(jpeg)
                    except Exception:
                        pass
                time.sleep(STREAM_INTERVAL)
        except KeyboardInterrupt:
            pass
    elif use_gui:
        try:
            while True:
                step_sim(1.0 / 60.0, SIM_HZ, blocking=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
