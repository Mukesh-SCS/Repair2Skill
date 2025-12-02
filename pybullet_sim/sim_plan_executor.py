"""Execute high-level repair-plan actions in the PyBullet scene.

This module translates plan steps (strings) into simple robot movements and
visual feedback. It handles both URDF-based parts (links) and procedural
parts (bodies) correctly.
"""

import pybullet as p
import json
import time
import math

# Use simple imports assuming this folder is in sys.path (set by app.py)
from sim_robot import move_ee, open_gripper, close_gripper
from sim_connection import step_sim

def load_json(path):
    """Load a JSON file from disk."""
    with open(path) as f:
        return json.load(f)


def get_pos(part_handle):
    """Return (pos, orn) for a part.
    
    Handles both (body, link) tuples and standalone body IDs.
    """
    if isinstance(part_handle, tuple):
        body, link = part_handle
        
        if link == -1:
            return p.getBasePositionAndOrientation(body)
            
        # Otherwise it's a child link of a larger URDF
        ls = p.getLinkState(body, link)
        return ls[0], ls[1]
        
    # Fallback if just an integer body ID is passed
    return p.getBasePositionAndOrientation(part_handle)


def recolor(part_handle, color):
    """Change the color of a part for visual feedback."""
    body, link = part_handle
    # changeVisualShape handles -1 correctly, so no special check needed
    p.changeVisualShape(body, link, rgbaColor=color)


def show_working_animation(robot, ee_link, parts, part_name):
    """Move robot to the part and wiggle it to simulate 'working'."""
    if part_name not in parts:
        print(f"Warning: {part_name} not found in scene")
        return

    try:
        target_pos, _ = get_pos(parts[part_name])
    except Exception as e:
        print(f"Warning: Could not get position of {part_name}: {e}")
        return
    
    try:
        # Move to hover above part - INCREASED STEPS for visibility
        hover_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.3]
        print(f"    Moving to hover position: {hover_pos}")
        move_ee(robot, ee_link, hover_pos, steps=80)  # Increased from 30
    except Exception as e:
        print(f"Warning: Could not move to hover position: {e}")
        step_sim(0.1)
        return
    
    try:
        # Move down to part - INCREASED STEPS
        work_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
        print(f"    Moving to work position: {work_pos}")
        move_ee(robot, ee_link, work_pos, steps=60)  # Increased from 20
    except Exception as e:
        print(f"Warning: Could not move to work position: {e}")
        step_sim(0.1)
        return
    
    # Wiggle action (simulate screwing/unscrewing) - MORE WIGGLES
    print(f"    Working on {part_name}...")
    for i in range(5):  # Increased from 3
        try:
            p.setJointMotorControl2(robot, ee_link, p.TORQUE_CONTROL, force=0)
        except:
            pass
        step_sim(0.1)  # Increased from 0.05
    
    try:
        # Return to hover - INCREASED STEPS
        print(f"    Returning to hover position")
        move_ee(robot, ee_link, hover_pos, steps=60)  # Increased from 20
    except Exception as e:
        print(f"Warning: Could not return to hover: {e}")
        step_sim(0.1)


def place_part(robot, ee_link, parts, part_name, gripper, open_val):
    """Teleport a part to the robot gripper and release it (simplified place)."""
    if part_name not in parts: return
    
    # 1. Open Gripper
    open_gripper(robot, gripper, open_val)
    
    # 2. Move EE to a drop zone above the chair
    drop_pos = [0.6, 0.0, 0.6]
    move_ee(robot, ee_link, drop_pos, steps=80)
    
    # 3. Teleport part to gripper (Cheating physics for stability)
    # We just make the part appear at the "correct" final location 
    # because 'placing' is very hard in physics sims without complex grasping.
    # For visual demo, we just ensure it is visible.
    pass 


def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step):
    """Execute a single repair step."""
    
    # Handle different JSON keys (GPT sometimes uses 'type', 'action_type', or 'action')
    action = step.get("type") or step.get("action_type") or step.get("action")
    part = step.get("target_part")

    if not action or not part:
        return

    action = action.lower()
    
    # Skip if part doesn't exist (e.g., 'generic' parts)
    if part not in parts and part != "":
        print(f"  [Skip] Part {part} not in visual scene")
        return

    if action == "inspect":
        print(f"  -> Inspecting {part}...")
        recolor(parts[part], (1, 1, 0, 1)) # Yellow
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.6, 0.4, 0.2, 1)) # Restore brown (simplification)
        step_sim(0.5)

    elif action == "remove":
        print(f"  -> Removing {part}...")
        show_working_animation(robot, ee_link, parts, part)
        # "Remove" by moving far away or making invisible
        body, link = parts[part]
        p.resetBasePositionAndOrientation(body, [10, 10, -10], [0,0,0,1])
        step_sim(0.5)

    elif action == "replace" or action == "attach":
        print(f"  -> Replacing/Attaching {part}...")
        # Teleport back to origin
        # (In a real app, you'd store the original pos, here we approximate)
        # Since our sim_scene sets positions statically, we can't easily undo "remove"
        # without reloading. For this demo, we assume "replace" just highlights it green.
        if part in parts:
             recolor(parts[part], (0, 1, 0, 1)) # Green
        show_working_animation(robot, ee_link, parts, part)
        step_sim(0.5)

    elif action == "tighten" or action == "fix":
        print(f"  -> Tightening {part}...")
        recolor(parts[part], (0, 0, 1, 1)) # Blue
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.6, 0.4, 0.2, 1)) # Restore
        step_sim(0.5)

    else:
        # Generic action
        print(f"  -> Processing {part} ({action})...")
        if part in parts:
            show_working_animation(robot, ee_link, parts, part)
        step_sim(0.5)