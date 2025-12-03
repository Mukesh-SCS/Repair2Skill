"""Simple CLI to run a repair plan inside the PyBullet scene.

This module wires together the connection, scene, robot, and plan executor
helpers so you can run a JSON-formatted repair plan and watch the simulated
robot perform each step.

Typical usage from the project root:
    python -m pybullet_sim.run_simulation --plan outputs/repair_plan_seat_loose.json
"""

import argparse
from sim_connection import connect, reset_camera, keep_window_open
from sim_robot import load_robot
from sim_scene import spawn_simple_chair
from sim_plan_executor import load_json, execute_step


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
    args = ap.parse_args()

    # Start a GUI PyBullet instance and position the camera for viewing.
    connect(gui=True)
    reset_camera()

    damaged = args.damaged_part

    # Load the robot model and spawn a simple chair with the specified
    # part marked as damaged (red color).

    robot, ee_link, gripper, open_val, close_val = load_robot(args.robot)
    parts = spawn_simple_chair(damaged)

    # Load and execute the repair plan step-by-step.

    plan = load_json(args.plan)
    seq = plan.get("repair_sequence", [])
    print("[INFO] Running", len(seq), "steps from plan:", args.plan)

    for step in seq:
        target_part = step.get("target_part", "")
       
        if target_part not in parts:
            print(f"[SKIP] Part '{target_part}' not in simulation, skipping step {step.get('step_id')}")
            continue
        execute_step(robot, ee_link, gripper, open_val, close_val, parts, step)

    # When finished keep the GUI open until the user closes it.
    keep_window_open()


if __name__ == "__main__":
    main()
