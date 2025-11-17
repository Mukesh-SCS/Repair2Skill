import argparse
from pybullet_sim.sim_connection import connect, reset_camera
from pybullet_sim.sim_robot import load_robot
from pybullet_sim.sim_scene import spawn_simple_chair
from pybullet_sim.sim_plan_executor import load_json, execute_step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--robot", default="kuka")
    args = ap.parse_args()

    connect(gui=True)
    reset_camera()

    damaged = "back_left_leg"
    robot, ee_link, gripper, open_val, close_val = load_robot(args.robot)
    parts = spawn_simple_chair(damaged)

    plan = load_json(args.plan)
    seq = plan.get("repair_sequence", [])
    print("[INFO] Running", len(seq), "steps")

    for step in seq:
        execute_step(robot, ee_link, gripper, open_val, close_val, parts, step)

if __name__ == "__main__":
    main()
