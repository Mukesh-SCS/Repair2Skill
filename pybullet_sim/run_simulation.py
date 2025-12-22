"""Simple CLI to run a repair plan inside the PyBullet scene.

This module wires together the connection, scene, robot, and plan executor
helpers so you can run a JSON-formatted repair plan and watch the simulated
robot perform each step.

Typical usage from the project root:
    python -m pybullet_sim.run_simulation --plan outputs/repair_plan_seat_loose.json
"""

import argparse
import os
import json
import threading
import time
from sim_connection import connect, reset_camera, keep_window_open, save_screenshot
from sim_robot import load_robot
from sim_scene import spawn_simple_chair
from sim_plan_executor import load_json, execute_step
try:
    from stream_server import start_streaming_server
except ImportError:
    start_streaming_server = None
    print("[WARN] stream_server not available, using legacy screenshot mode", flush=True)


def main():
    """Parse CLI args, set up the simulation, and execute the plan.

    Arguments supported mirror the simple demo needs: which plan to run,
    an optional repair graph (not required by executor), robot type, and
    which chair part should be marked as damaged for visualization.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to repair plan JSON")
    ap.add_argument("--graph", default=None, help="(Optional) repair graph JSON")
    ap.add_argument(
        "--robot",
        default="kuka",
        help="Robot type identifier for sim_robot.load_robot"
    )
    ap.add_argument(
        "--damaged-part",
        default="back_left_leg",
        help="Name of the damaged chair part to highlight in the scene"
    )
    ap.add_argument("--camera-dist", type=float, default=1.8, help="Camera distance")
    ap.add_argument("--camera-yaw", type=float, default=40, help="Camera yaw")
    ap.add_argument("--camera-pitch", type=float, default=-35, help="Camera pitch")
    ap.add_argument("--camera-params", help="Path to JSON file with dynamic camera parameters")
    ap.add_argument("--screenshot", help="Path to save screenshot (legacy, use --stream-port instead)")
    ap.add_argument("--stream-port", type=int, default=8080, help="Port for direct frame streaming (default: 8080)")
    ap.add_argument("--gui", action="store_true", help="Enable PyBullet GUI window (default: headless)")
    args = ap.parse_args()

    # Flush output immediately for better logging
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    print(f"[INFO] Starting simulation with plan: {args.plan}", flush=True)
    print(f"[INFO] Screenshot path: {args.screenshot}", flush=True)
    print(f"[INFO] Camera params: {args.camera_params}", flush=True)

    # Verify plan file exists
    if not os.path.exists(args.plan):
        print(f"[ERROR] Plan file not found: {args.plan}", flush=True)
        sys.exit(1)

    try:
        # Start a PyBullet instance (GUI or headless)
        print("[INFO] Connecting to PyBullet...", flush=True)
        connect(gui=args.gui)
        print("[INFO] PyBullet connected successfully", flush=True)
        reset_camera(dist=args.camera_dist, yaw=args.camera_yaw, pitch=args.camera_pitch)
        print("[INFO] Camera reset", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to connect to PyBullet: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    damaged = args.damaged_part

    # Load the robot model and spawn a simple chair with the specified
    # part marked as damaged (red color).

    try:
        print(f"[INFO] Loading robot: {args.robot}", flush=True)
        robot, ee_link, gripper, open_val, close_val = load_robot(args.robot)
        print("[INFO] Robot loaded successfully", flush=True)
        print(f"[INFO] Spawning chair with damaged part: {damaged}", flush=True)
        parts = spawn_simple_chair(damaged)
        print(f"[INFO] Chair spawned with {len(parts)} parts", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load robot or spawn chair: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Capture original positions for replacement logic
    from sim_plan_executor import get_pos
    original_positions = {}
    for part_name, part_handle in parts.items():
        try:
            pos, _ = get_pos(part_handle)
            original_positions[part_name] = pos
        except Exception as e:
            print(f"[WARN] Could not get position for {part_name}: {e}")

    # Start direct frame streaming server (better than screenshots!)
    stream_server = None
    if args.stream_port and start_streaming_server:
        try:
            camera_params_path = args.camera_params if args.camera_params else None
            stream_server = start_streaming_server(args.stream_port, camera_params_path)
            print(f"[INFO] Direct frame streaming enabled on port {args.stream_port}", flush=True)
            print(f"[INFO] Access stream at: http://localhost:{args.stream_port}/frame.jpg", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to start streaming server: {e}", flush=True)
            print(f"[INFO] Falling back to legacy screenshot mode", flush=True)
            stream_server = None
    
    # Legacy screenshot support (for backward compatibility)
    screenshot_path = None
    if args.screenshot:
        screenshot_path = os.path.abspath(args.screenshot)
        screenshot_dir = os.path.dirname(screenshot_path)
        os.makedirs(screenshot_dir, exist_ok=True)
        print(f"[INFO] Legacy screenshot mode enabled: {screenshot_path}", flush=True)

    # Load and execute the repair plan step-by-step.
    try:
        print(f"[INFO] Loading repair plan from: {args.plan}", flush=True)
        plan = load_json(args.plan)
        seq = plan.get("repair_sequence", [])
        print(f"[INFO] Running {len(seq)} steps from plan: {args.plan}", flush=True)
        if not seq:
            print("[WARN] No repair steps found in plan!", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load plan: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    for step in seq:
        target_part = step.get("target_part", "")
       
        if target_part not in parts:
            print(f"[SKIP] Part '{target_part}' not in simulation, skipping step {step.get('step_id')}")
            continue
        execute_step(robot, ee_link, gripper, open_val, close_val, parts, step, original_positions=original_positions)
        
        # Legacy screenshot support (if enabled)
        if screenshot_path:
            try:
                cam_dist = args.camera_dist
                cam_yaw = args.camera_yaw
                cam_pitch = args.camera_pitch
                
                if args.camera_params and os.path.exists(args.camera_params):
                    try:
                        with open(args.camera_params, 'r') as f:
                            cam_params = json.load(f)
                            cam_dist = cam_params.get('dist', cam_dist)
                            cam_yaw = cam_params.get('yaw', cam_yaw)
                            cam_pitch = cam_params.get('pitch', cam_pitch)
                    except Exception:
                        pass
                
                save_screenshot(
                    screenshot_path, 
                    width=640, 
                    height=480,
                    dist=cam_dist,
                    yaw=cam_yaw,
                    pitch=cam_pitch,
                    target=[0.6, 0.0, 0.4]
                )
            except Exception as e:
                print(f"[WARN] Failed to save step screenshot: {e}")

    # Keep simulation running continuously for live streaming
    print("[INFO] Repair plan execution complete. Keeping simulation running for live streaming...")
    
    if stream_server:
        print("[INFO] Direct frame streaming active. Use camera controls to view from different angles.")
        print(f"[INFO] Stream available at: http://localhost:{args.stream_port}/frame.jpg")
        try:
            # Keep the simulation alive - streaming server handles frames
            import pybullet as p
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 240)  # 240 Hz physics update rate
        except KeyboardInterrupt:
            print("[INFO] Simulation stopped by user.")
            stream_server.shutdown()
            import pybullet as p
            p.disconnect()
    elif screenshot_path:
        # Legacy screenshot mode
        print("[INFO] Legacy screenshot mode active.")
        try:
            import pybullet as p
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 240)
        except KeyboardInterrupt:
            print("[INFO] Simulation stopped by user.")
            import pybullet as p
            p.disconnect()
    else:
        # If no streaming, just keep window open if GUI mode
        if args.gui:
            keep_window_open()
        else:
            import pybullet as p
            p.disconnect()


if __name__ == "__main__":
    main()
