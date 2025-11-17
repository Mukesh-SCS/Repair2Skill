import pybullet as p
from .sim_robot import move_ee, open_gripper, close_gripper
from .sim_connection import step_sim
import json

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ------------ LLM ACTION INTERPRETER ---------------- #

def classify_action(action: str):
    a = action.lower()

    if any(k in a for k in ["remove","detach","pull","take off","disassemble"]):
        return "remove"
    if any(k in a for k in ["attach","replace","install","put back","assemble"]):
        return "attach"
    if "tighten" in a or "screw" in a:
        return "tighten"
    if "inspect" in a or "check" in a:
        return "inspect"
    if any(k in a for k in ["clean","wipe","sand"]):
        return "clean"
    if any(k in a for k in ["align","position"]):
        return "align"

    return "generic"

# ------------ EXECUTION ROUTINES ---------------- #

def move_to_part(robot, ee_link, parts, part):
    pos, _ = p.getBasePositionAndOrientation(parts[part])
    hover = [pos[0], pos[1], pos[2] + 0.20]
    move_ee(robot, ee_link, hover)

def pick_part(robot, ee_link, parts, part, gripper, close_val):
    pos,_ = p.getBasePositionAndOrientation(parts[part])
    move_ee(robot, ee_link, [pos[0], pos[1], pos[2] + 0.05])
    if gripper:
        close_gripper(robot, gripper, close_val)
    step_sim()

def place_part(robot, ee_link, parts, part, gripper, open_val):
    pos,_ = p.getBasePositionAndOrientation(parts[part])
    move_ee(robot, ee_link, [pos[0], pos[1], pos[2] + 0.05])
    if gripper:
        open_gripper(robot, gripper, open_val)
    step_sim()

# color changes
def recolor(part_id, color):
    p.changeVisualShape(part_id, -1, rgbaColor=color)

# ------------ MAIN STEP EXECUTION ---------------- #

def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step):
    action = classify_action(step["action"])
    part = step["target_part"]

    print(f"[EXECUTE] Step {step['step_id']}: {step['action']} → {part} ({action})")

    move_to_part(robot, ee_link, parts, part)

    if action == "remove":
        pick_part(robot, ee_link, parts, part, gripper, close_val)
        recolor(parts[part], (0.9,0.6,0.3,1))

    elif action == "attach":
        place_part(robot, ee_link, parts, part, gripper, open_val)
        recolor(parts[part], (0.3,0.8,0.4,1))

    elif action == "tighten":
        step_sim(0.3)

    elif action == "inspect":
        step_sim(0.4)

    elif action == "clean":
        step_sim(0.3)

    elif action == "align":
        step_sim(0.3)

    else:
        step_sim(0.3)  # fallback
