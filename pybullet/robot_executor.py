"""
Repair2Skill – Stage 5: PyBullet Simulation Executor
Runs a repair plan produced by GPT and follows a repair graph order.

Inputs
  outputs/repair_plan_<part>_<damage>.json
  outputs/repair_graph_<part>.json  (optional)

Usage
  python scripts/robot_executor.py
  python scripts/robot_executor.py --plan outputs/repair_plan_seat_broken.json --graph outputs/repair_graph_seat.json

Notes
- Uses KUKA iiwa by default. Switch to Panda with --robot panda.
- Scene uses simple blocks as a stand-in chair so you can run now.
"""

import os
import json
import time
import math
import argparse
from pathlib import Path

import pybullet as p
import pybullet_data

# ------------------------------- sim utils -------------------------------- #

def connect(gui=True):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    return cid

def load_robot(robot="kuka"):
    if robot == "panda":
        rid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 11  # Panda hand
        gripper_joints = [9, 10]  # finger joints
        open_val, close_val = 0.04, 0.0
    else:
        rid = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 6   # last link index for KUKA iiwa
        gripper_joints = []  # no gripper in default URDF
        open_val, close_val = None, None
    return rid, ee_link, gripper_joints, open_val, close_val

def reset_camera(target=[0.6, 0.0, 0.4], dist=2.0, yaw=45, pitch=-30):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw,
                                 cameraPitch=pitch, cameraTargetPosition=target)

def step_sim(seconds=1.0, hz=240):
    for _ in range(int(seconds * hz)):
        p.stepSimulation()
        time.sleep(1.0 / hz)

# ------------------------------- scene setup ------------------------------- #

def create_block(size=(0.2, 0.2, 0.02), pos=(0.6, 0.0, 0.1), rgba=(0.75, 0.75, 0.75, 1)):
    """Creates a visual+collision box and returns its body id."""
    half = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=pos)
    return bid

def spawn_simple_chair(damaged_part="back_left_leg"):
    """
    Builds a very simple 'chair' with separate links as blocks:
      - seat
      - back
      - four legs: front_left_leg, front_right_leg, back_left_leg, back_right_leg
    Returns dict mapping part names -> body ids and default poses.
    """
    base_x, base_y = 0.6, 0.0
    parts = {}

    # seat
    parts["seat"] = create_block(size=(0.4, 0.4, 0.04), pos=(base_x, base_y, 0.42), rgba=(0.8, 0.8, 0.8, 1))
    # backrest
    parts["back"] = create_block(size=(0.4, 0.06, 0.4), pos=(base_x, base_y - 0.22, 0.64), rgba=(0.85, 0.85, 0.85, 1))

    # legs
    leg_h = 0.42
    leg_w = 0.06
    offsets = {
        "front_left_leg":  ( base_x - 0.16, base_y + 0.16, leg_h/2 ),
        "front_right_leg": ( base_x + 0.16, base_y + 0.16, leg_h/2 ),
        "back_left_leg":   ( base_x - 0.16, base_y - 0.16, leg_h/2 ),
        "back_right_leg":  ( base_x + 0.16, base_y - 0.16, leg_h/2 ),
    }
    for name, (x, y, z) in offsets.items():
        color = (1, 0.3, 0.3, 1) if name == damaged_part else (0.7, 0.7, 0.7, 1)
        parts[name] = create_block(size=(leg_w, leg_w, leg_h), pos=(x, y, z), rgba=color)

    return parts

def get_body_pose(body_id):
    pos, orn = p.getBasePositionAndOrientation(body_id)
    return pos, orn

# ---------------------------- motion primitives ---------------------------- #

def open_gripper(robot, gripper_joints, open_val):
    if not gripper_joints:
        return
    for j in gripper_joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, open_val, force=50)

def close_gripper(robot, gripper_joints, close_val):
    if not gripper_joints:
        return
    for j in gripper_joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, close_val, force=50)

def move_ee_ik(robot, ee_link, target_pos, target_orn=None, steps=120):
    """
    Simple IK to move the end-effector near a target position.
    target_orn: quaternion or None (will keep current).
    """
    if target_orn is None:
        _, target_orn = p.getLinkState(robot, ee_link)[5], p.getLinkState(robot, ee_link)[6]

    joint_positions = p.calculateInverseKinematics(robot, ee_link, target_pos, target_orn,
                                                   maxNumIterations=200, residualThreshold=1e-3)
    num_joints = p.getNumJoints(robot)
    for i in range(min(len(joint_positions), num_joints)):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_positions[i], force=200)

    step_sim(steps / 240.0)

def wiggle_wrist(robot, ee_link, angle=0.5):
    """Simulate tightening by small oscillation on final joints."""
    n = p.getNumJoints(robot)
    if n < 2:
        return
    j = n - 1
    base = p.getJointState(robot, j)[0]
    for _ in range(3):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, base + angle, force=50)
        step_sim(0.3)
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, base - angle, force=50)
        step_sim(0.3)
    p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, base, force=50)

def attach_constraint(holder_body, target_body, offset=(0,0,0.02)):
    """Fake grasp: create a fixed constraint to 'hold' an object."""
    holder_pos, holder_orn = p.getBasePositionAndOrientation(holder_body)
    tgt_pos, tgt_orn = p.getBasePositionAndOrientation(target_body)
    cid = p.createConstraint(
        parentBodyUniqueId=holder_body,
        parentLinkIndex=-1,
        childBodyUniqueId=target_body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=offset,
        childFramePosition=[0, 0, 0],
    )
    return cid

# ------------------------------- plan handling ----------------------------- #

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_latest_plan():
    outs = Path("outputs")
    if not outs.exists():
        return None
    plans = sorted([p for p in outs.glob("repair_plan_*.json")])
    return str(plans[-1]) if plans else None

def infer_graph_from_plan(plan_path):
    name = Path(plan_path).name  # repair_plan_<part>_<damage>.json
    core = name.replace("repair_plan_", "")
    part = core.split("_")[0]
    cand = Path("outputs") / f"repair_graph_{part}.json"
    return str(cand) if cand.exists() else None, part

def dependency_order(graph_dict):
    """Return bottom-up order from a tree dict {parent: [child,...]}."""
    order, seen = [], set()
    def dfs(node):
        if node in seen:
            return
        seen.add(node)
        for c in graph_dict.get(node, []):
            dfs(c)
        order.append(node)
    for root in list(graph_dict.keys()):
        dfs(root)
    return order

def plan_steps_in_dep_order(plan, graph):
    if not graph:
        return plan.get("repair_sequence", [])
    dep = dependency_order(graph)
    seq = plan.get("repair_sequence", [])
    # keep only steps whose target_part appears in graph, sorted by dependency
    dep_index = {p: i for i, p in enumerate(dep)}
    seq = [s for s in seq if s.get("target_part") in dep_index]
    return sorted(seq, key=lambda s: dep_index[s["target_part"]])

# ------------------------------- action map -------------------------------- #

def target_pos_for_part(parts_dict, part_name, lift=0.15):
    """Return a grasp approach position slightly above the part center."""
    body = parts_dict.get(part_name)
    if body is None:
        return None
    (x, y, z), _ = p.getBasePositionAndOrientation(body)
    return [x, y, z + lift]

def execute_step(robot, ee_link, gripper_joints, open_val, close_val, parts, step):
    action = step.get("action", "").lower()
    part = step.get("target_part", "")
    tool = step.get("tool", "gripper")
    print(f"[STEP {step.get('step_id','?')}] {action} → {part} ({tool})")

    # 1) move above target
    tgt = target_pos_for_part(parts, part, lift=0.20)
    if tgt:
        move_ee_ik(robot, ee_link, tgt, steps=160)

    # 2) descend
    if tgt:
        descend = [tgt[0], tgt[1], tgt[2] - 0.18]
        move_ee_ik(robot, ee_link, descend, steps=160)

    # 3) emulate grasp if available
    if gripper_joints:
        close_gripper(robot, gripper_joints, close_val)

    # 4) action-specific effect
    if action in ("remove", "detach"):
        # lift the part up to show removal
        if tgt:
            lift = [tgt[0], tgt[1], tgt[2] + 0.10]
            move_ee_ik(robot, ee_link, lift, steps=160)
            # recolor to indicate removed
            pid = parts.get(part)
            if pid is not None:
                p.changeVisualShape(pid, -1, rgbaColor=(0.9, 0.6, 0.3, 1))
    elif action in ("replace", "attach"):
        # recolor to "fixed" green
        pid = parts.get(part)
        if pid is not None:
            p.changeVisualShape(pid, -1, rgbaColor=(0.3, 0.8, 0.4, 1))
    elif action in ("tighten", "screw", "torque"):
        wiggle_wrist(robot, ee_link, angle=0.6)
    elif action in ("inspect", "verify"):
        step_sim(0.5)

    # 5) retreat
    if tgt:
        retreat = [tgt[0], tgt[1], tgt[2] + 0.25]
        move_ee_ik(robot, ee_link, retreat, steps=160)

    print(f"✓ {action} {part} done.")

# ---------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, default=None, help="Path to repair_plan_*.json")
    parser.add_argument("--graph", type=str, default=None, help="Path to repair_graph_*.json")
    parser.add_argument("--robot", type=str, default="kuka", choices=["kuka", "panda"])
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan or find_latest_plan()
    if not plan_path or not os.path.exists(plan_path):
        print("[ERROR] No repair plan found. Generate it with main.py first.")
        return

    graph_path = args.graph
    if graph_path is None:
        graph_path, damaged_part_from_name = infer_graph_from_plan(plan_path)
    else:
        damaged_part_from_name = None

    plan = load_json(plan_path)
    graph = load_json(graph_path) if graph_path and os.path.exists(graph_path) else {}

    # If no graph, try to infer damaged part from the first step
    damaged_part = damaged_part_from_name
    if not damaged_part:
        seq = plan.get("repair_sequence", [])
        damaged_part = seq[0]["target_part"] if seq else "back_left_leg"

    cid = connect(gui=not args.headless)
    robot, ee_link, gripper_joints, open_val, close_val = load_robot(args.robot)
    reset_camera()

    # Build a quick chair scene and color the damaged part red
    parts = spawn_simple_chair(damaged_part=damaged_part)

    # Open gripper if available
    open_gripper(robot, gripper_joints, open_val)
    step_sim(0.5)

    # Order steps with dependency graph if present
    seq = plan_steps_in_dep_order(plan, graph)
    if graph:
        order = dependency_order(graph)
        print("[INFO] Dependency order:", " -> ".join(order))

    # Execute plan
    for step in seq:
        execute_step(robot, ee_link, gripper_joints, open_val, close_val, parts, step)
        step_sim(0.3)

    print("[INFO] Simulation complete. Close the window to exit.")
    if args.headless:
        p.disconnect()

if __name__ == "__main__":
    main()
