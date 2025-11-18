import pybullet as p
from sim_robot import move_ee, open_gripper, close_gripper
from sim_connection import step_sim
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

# ------------ VISUAL FEEDBACK ---------------------#

def highlight_part(part_id, is_working=False):
    """Highlight a part - bright color if being worked on"""
    if is_working:
        # Bright yellow for active work
        p.changeVisualShape(part_id, -1, rgbaColor=(1, 1, 0, 1))
    else:
        # Normal state
        p.changeVisualShape(part_id, -1, rgbaColor=(0.7, 0.7, 0.7, 1))

def recolor(part_id, color):
    """Change part color"""
    p.changeVisualShape(part_id, -1, rgbaColor=color)

def show_working_animation(robot, ee_link, parts, part, duration=0.5):
    """Animate the robot working on a part"""
    pos, _ = p.getBasePositionAndOrientation(parts[part])
    
    # Highlight the part being worked on
    recolor(parts[part], (1, 1, 0, 1))  # Yellow = being worked on
    
    # Simulate working motion (up and down)
    for i in range(10):
        work_pos = [pos[0], pos[1], pos[2] + 0.05 + (i % 2) * 0.02]
        move_ee(robot, ee_link, work_pos)
    
    # Restore part to normal color
    recolor(parts[part], (0.7, 0.7, 0.7, 1))  # Gray = done

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

# ------------ MAIN STEP EXECUTION ---------------- #

def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step):
    action = classify_action(step["action"])
    part = step["target_part"]

    print(f"[EXECUTE] Step {step['step_id']}: {step['action']} -> {part} ({action})")
    
    # Highlight part being worked on
    if part in parts:
        recolor(parts[part], (1, 1, 0, 1))  # Yellow highlight
    
    move_to_part(robot, ee_link, parts, part)

    if action == "remove":
        print(f"  -> Removing {part}...")
        pick_part(robot, ee_link, parts, part, gripper, close_val)
        recolor(parts[part], (0.9, 0.6, 0.3, 1))  # Orange when removed
        step_sim(0.3)

    elif action == "attach":
        print(f"  -> Attaching {part}...")
        place_part(robot, ee_link, parts, part, gripper, open_val)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green when attached/fixed
        step_sim(0.3)

    elif action == "tighten":
        print(f"  -> Tightening {part}...")
        show_working_animation(robot, ee_link, parts, part, 0.3)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green when fixed
        step_sim(0.3)

    elif action == "inspect":
        print(f"  -> Inspecting {part}...")
        show_working_animation(robot, ee_link, parts, part, 0.2)
        recolor(parts[part], (0.7, 0.7, 0.7, 1))  # Gray after inspection
        step_sim(0.4)

    elif action == "clean":
        print(f"  -> Cleaning {part}...")
        show_working_animation(robot, ee_link, parts, part, 0.3)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green when cleaned
        step_sim(0.3)

    elif action == "align":
        print(f"  -> Aligning {part}...")
        show_working_animation(robot, ee_link, parts, part, 0.3)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green when aligned
        step_sim(0.3)

    else:
        print(f"  -> Processing {part}...")
        show_working_animation(robot, ee_link, parts, part, 0.3)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green when done
        step_sim(0.3)
    
    # Final state: reset to normal unless part is damaged/repaired
    if part in parts and action not in ["remove"]:
        step_sim(0.5)  # Pause to show completed state
