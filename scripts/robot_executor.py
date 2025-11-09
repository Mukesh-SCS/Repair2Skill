"""
================================================================================
Repair2Skill Stage 4: Robotic Repair Execution with PyBullet
================================================================================
Reads:
  - outputs/repair_plan_<part>_<damage>.json
  - outputs/repair_graph_<part>.json

Executes the repair sequence in PyBullet following dependency order.
================================================================================
"""

import os
import json
import time
import pybullet as p
import pybullet_data


# --------------------------------------------------------------------------
#  Setup Simulation
# --------------------------------------------------------------------------

def setup_simulation(gui=True):
    """Initialize PyBullet environment."""
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    return cid


def load_models():
    """Load robot arm and placeholder furniture model."""
    robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    # Replace "table.urdf" with your custom chair model if available
    furniture = p.loadURDF("table/table.urdf", [0.6, 0, 0])
    return robot, furniture


# --------------------------------------------------------------------------
#  Action Execution
# --------------------------------------------------------------------------

def execute_motion(robot, step_name, joint_targets=None, duration=1.0):
    """Simple motion primitive to simulate action."""
    print(f"[ACTION] Executing motion for: {step_name}")
    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        target = 0.3 if joint_targets is None else joint_targets.get(j, 0.3)
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=target)
    for _ in range(int(240 * duration)):
        p.stepSimulation()
        time.sleep(1 / 240.0)


def execute_repair_step(robot, step, dependencies_done):
    """Simulate a single repair step considering dependencies."""
    step_id = step.get("step_id", "?")
    action = step.get("action", "inspect").upper()
    part = step.get("target_part", "unknown")
    tool = step.get("tool", "gripper")

    # Skip if dependent part not yet processed
    if part not in dependencies_done:
        print(f"[WAIT] Dependencies not satisfied for {part}.")
        return False

    print(f"[STEP {step_id}] {action} → {part} using {tool}")
    execute_motion(robot, f"{action.lower()}_{part}")
    print(f"✅ Completed step {step_id}: {action} {part}")
    return True


# --------------------------------------------------------------------------
#  Dependency Management
# --------------------------------------------------------------------------

def load_repair_graph(graph_path):
    """Load repair dependency graph from JSON."""
    if not os.path.exists(graph_path):
        print(f"[WARN] No repair graph found at {graph_path}. Using flat order.")
        return {}
    with open(graph_path, "r") as f:
        return json.load(f)


def dependency_order(graph):
    """Return list of parts sorted by dependency order (bottom-up)."""
    order, visited = [], set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for child in graph.get(node, []):
            dfs(child)
        order.append(node)

    for node in list(graph.keys()):
        dfs(node)
    return order


# --------------------------------------------------------------------------
#  Main Simulation
# --------------------------------------------------------------------------

def run_repair_simulation(plan_path, graph_path=None, gui=True):
    """Run repair simulation following dependency order."""
    if not os.path.exists(plan_path):
        print(f"[ERROR] Repair plan not found: {plan_path}")
        return

    with open(plan_path, "r") as f:
        plan = json.load(f)

    steps = plan.get("repair_sequence", [])
    if not steps:
        print("[WARN] No repair steps in JSON.")
        return

    # Load dependency graph
    graph = load_repair_graph(graph_path) if graph_path else {}
    dep_order = dependency_order(graph)
    print(f"[INFO] Dependency execution order: {dep_order}")

    cid = setup_simulation(gui=gui)
    robot, furniture = load_models()

    processed_parts = set(dep_order) if not dep_order else set()

    for step in steps:
        part = step.get("target_part", "")
        if dep_order and part not in dep_order:
            print(f"[SKIP] {part} not in repair graph.")
            continue
        ok = execute_repair_step(robot, step, processed_parts)
        if ok:
            processed_parts.add(part)

    print("[INFO] All repair steps executed. Disconnecting...")
    p.disconnect()


# --------------------------------------------------------------------------
#  Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Automatically find latest repair plan
    plan_files = [f for f in os.listdir("outputs") if f.startswith("repair_plan_")]
    if not plan_files:
        print("[ERROR] No repair plans found in outputs/.")
        exit()

    plan_path = os.path.join("outputs", sorted(plan_files)[-1])
    part_name = plan_path.split("repair_plan_")[-1].split("_")[0]
    graph_path = f"outputs/repair_graph_{part_name}.json"

    print(f"[INFO] Using plan: {plan_path}")
    print(f"[INFO] Using graph: {graph_path}")

    run_repair_simulation(plan_path, graph_path, gui=True)
