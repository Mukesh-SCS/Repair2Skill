import argparse, os
from sim_connection import connect, reset_camera
from sim_robot import load_robot, open_gripper
from sim_scene import spawn_simple_chair
from sim_plan_executor import load_json, execute_step, dependency_order
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, default="outputs/repair_plan_back_left_leg_broken.json")
    parser.add_argument("--graph", type=str, default=None)
    args = parser.parse_args()

    connect(gui=True)
    robot, ee_link, gripper_joints, open_val, close_val = load_robot("kuka")
    reset_camera()
    parts = spawn_simple_chair("back_left_leg")
    plan = load_json(args.plan)
    graph = load_json(args.graph) if args.graph and os.path.exists(args.graph) else {}

    seq = plan.get("repair_sequence", [])
    print("[INFO] Executing", len(seq), "steps...")
    for s in seq:
        execute_step(robot, ee_link, gripper_joints, open_val, close_val, parts, s)

if __name__ == "__main__":
    main()
