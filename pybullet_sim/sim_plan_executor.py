import pybullet as p
from sim_robot import move_ee, open_gripper, close_gripper
from sim_connection import step_sim
import json

def load_json(path):
    with open(path) as f:
        return json.load(f)


# ================================
# Utilities for body/link handling
# ================================

def get_pos(part_handle):
    """Return world position of a part, whether it's a body or (body, link)."""
    if isinstance(part_handle, tuple):
        body, link = part_handle
        ls = p.getLinkState(body, link)
        return ls[0], ls[1]  # pos, orn
    else:
        return p.getBasePositionAndOrientation(part_handle)


def change_color(part_handle, color):
    """Unified color setter for body or (body, link)."""
    if isinstance(part_handle, tuple):
        body, link = part_handle
        p.changeVisualShape(body, link, rgbaColor=color)
    else:
        p.changeVisualShape(part_handle, -1, rgbaColor=color)


# ===========================
# LLM ACTION INTERPRETER
# ===========================

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


# ===========================
# VISUAL FEEDBACK
# ===========================

def highlight_part(part_handle, is_working=False):
    """Highlight a part."""
    if is_working:
        change_color(part_handle, (1, 1, 0, 1))  # Yellow
    else:
        change_color(part_handle, (0.7, 0.7, 0.7, 1))  # Gray


def recolor(part_handle, color):
    change_color(part_handle, color)


def show_working_animation(robot, ee_link, parts, part, duration=0.5):
    pos, _ = get_pos(parts[part])

    # Highlight part
    recolor(parts[part], (1, 1, 0, 1))

    # Up-down motion
    for i in range(10):
        work_pos = [pos[0], pos[1], pos[2] + 0.05 + (i % 2) * 0.02]
        move_ee(robot, ee_link, work_pos)

    # Restore normal color
    recolor(parts[part], (0.7, 0.7, 0.7, 1))


# ===========================
# MOVEMENT ROUTINES
# ===========================

def move_to_part(robot, ee_link, parts, part):
    pos, _ = get_pos(parts[part])
    hover = [pos[0], pos[1], pos[2] + 0.20]
    move_ee(robot, ee_link, hover)


def pick_part(robot, ee_link, parts, part, gripper, close_val):
    pos, _ = get_pos(parts[part])
    move_ee(robot, ee_link, [pos[0], pos[1], pos[2] + 0.05])
    if gripper:
        close_gripper(robot, gripper, close_val)
    step_sim()


def place_part(robot, ee_link, parts, part, gripper, open_val):
    pos, _ = get_pos(parts[part])
    move_ee(robot, ee_link, [pos[0], pos[1], pos[2] + 0.05])
    if gripper:
        open_gripper(robot, gripper, open_val)
    step_sim()


# ===========================
# MAIN EXECUTION
# ===========================

def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step):
    action = classify_action(step["action"])
    part = step["target_part"]

    print(f"[EXECUTE] Step {step['step_id']}: {step['action']} -> {part} ({action})")

    # Highlight active part
    if part in parts:
        recolor(parts[part], (1, 1, 0, 1))

    move_to_part(robot, ee_link, parts, part)

    if action == "remove":
        print(f"  -> Removing {part}...")
        pick_part(robot, ee_link, parts, part, gripper, close_val)
        recolor(parts[part], (0.9, 0.6, 0.3, 1))  # Orange
        step_sim(0.3)

    elif action == "attach":
        print(f"  -> Attaching {part}...")
        place_part(robot, ee_link, parts, part, gripper, open_val)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green
        step_sim(0.3)

    elif action == "tighten":
        print(f"  -> Tightening {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    elif action == "inspect":
        print(f"  -> Inspecting {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.7, 0.7, 0.7, 1))
        step_sim(0.4)

    elif action == "clean":
        print(f"  -> Cleaning {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    elif action == "align":
        print(f"  -> Aligning {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    else:
        print(f"  -> Processing {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    # Pause after step
    if part in parts and action != "remove":
        step_sim(0.5)
