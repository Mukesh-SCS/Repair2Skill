import json, os
from pathlib import Path
from .sim_robot import move_ee_ik, close_gripper, open_gripper
from .sim_connection import step_sim

def load_json(path):
    with open(path) as f: return json.load(f)

def dependency_order(graph):
    order, seen = [], set()
    def dfs(node):
        if node in seen: return
        seen.add(node)
        for c in graph.get(node, []): dfs(c)
        order.append(node)
    for r in graph: dfs(r)
    return order

def execute_step(robot, ee_link, gripper_joints, open_val, close_val, parts, step):
    action = step["action"].lower()
    part = step["target_part"]
    print(f"[STEP {step['step_id']}] {action} → {part}")
    tgt = [0.6, 0.0, 0.5]
    move_ee_ik(robot, ee_link, tgt)
    if action in ("remove", "detach"):
        p.changeVisualShape(parts[part], -1, rgbaColor=(0.9, 0.6, 0.3, 1))
    elif action in ("replace", "attach"):
        p.changeVisualShape(parts[part], -1, rgbaColor=(0.3, 0.8, 0.4, 1))
    elif action in ("tighten",):
        step_sim(0.5)
    step_sim(0.3)
