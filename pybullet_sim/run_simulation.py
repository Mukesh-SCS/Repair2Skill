import argparse
from sim_connection import connect, reset_camera, keep_window_open
from sim_robot import load_robot
from sim_scene import spawn_simple_chair
from sim_plan_executor import load_json, execute_step


def main():
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

    connect(gui=True)
    reset_camera()

    damaged = args.damaged_part
    robot, ee_link, gripper, open_val, close_val = load_robot(args.robot)
    parts = spawn_simple_chair(damaged)

    plan = load_json(args.plan)
    seq = plan.get("repair_sequence", [])
    print("[INFO] Running", len(seq), "steps from plan:", args.plan)

    for step in seq:
        target_part = step.get("target_part", "")
        # Skip if part not in simulation
        if target_part not in parts:
            print(f"[SKIP] Part '{target_part}' not in simulation, skipping step {step.get('step_id')}")
            continue
        execute_step(robot, ee_link, gripper, open_val, close_val, parts, step)

    # Keep window open until user closes it
    keep_window_open()


if __name__ == "__main__":
    main()
