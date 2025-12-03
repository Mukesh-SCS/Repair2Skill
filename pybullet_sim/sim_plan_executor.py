"""Execute high-level repair-plan actions in the PyBullet scene.

This module translates plan steps (strings) into simple robot movements and
visual feedback. It handles both URDF-based parts (links) and procedural
parts (bodies) correctly.
"""

import pybullet as p
import json
import time
import math
from sim_robot import move_ee, open_gripper, close_gripper
from sim_connection import step_sim
import time

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
            
        
        ls = p.getLinkState(body, link)
        return ls[0], ls[1]
        
    
    return p.getBasePositionAndOrientation(part_handle)


def recolor(part_handle, color):
    """Change the color of a part for visual feedback."""
    body, link = part_handle
    p.changeVisualShape(body, link, rgbaColor=color)


def safe_call(fn, *args, timeout=12, **kwargs):
    """Call `fn` and warn if it takes longer than `timeout` seconds.

    This avoids the simulation worker appearing to freeze when IK or
    motion steps unexpectedly block for long periods.
    """
    t0 = time.time()
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[SIM] safe_call exception in {fn.__name__}: {e}")
        return False
    dt = time.time() - t0
    if dt > timeout:
        print(f"[SIM] safe_call: {fn.__name__} took {dt:.1f}s (> {timeout}s). Continuing.")
    return True


def show_working_animation(robot, ee_link, parts, part_name, original_positions=None):
    """Move robot to the part and wiggle it to simulate 'working'."""
    if part_name not in parts:
        print(f"Warning: {part_name} not found in scene")
        return

    try:
        target_pos, _ = get_pos(parts[part_name])
        # If the part has been teleported away (e.g. removed), fallback to the
        # originally stored position so the robot doesn't travel to the sentinel
        # teleport coordinates used by the "remove" action.
        invalid = False
        if target_pos is None:
            invalid = True
        else:
            tx, ty, tz = target_pos
            if abs(tx) > 5.0 or abs(ty) > 5.0 or tz < -1.0:
                invalid = True

        if invalid and original_positions and part_name in original_positions:
            stored = original_positions[part_name]
            print(f"    [SIM] Part '{part_name}' appears teleported; using original pos {stored}")
            target_pos = stored
    except Exception as e:
        print(f"Warning: Could not get position of {part_name}: {e}")
        return
    
    try:
       
        hover_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.3]
        print(f"    Moving to hover position: {hover_pos}")
        move_ee(robot, ee_link, hover_pos, steps=80)  
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
    
    pass 


def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step, original_positions=None):
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
        safe_call(show_working_animation, robot, ee_link, parts, part, original_positions=original_positions, timeout=12)
        recolor(parts[part], (0.6, 0.4, 0.2, 1)) # Restore brown (simplification)
        step_sim(0.5)

    elif action == "remove":
        print(f"  -> Removing {part}...")
        safe_call(show_working_animation, robot, ee_link, parts, part, original_positions=original_positions, timeout=12)
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
        safe_call(show_working_animation, robot, ee_link, parts, part, original_positions=original_positions, timeout=12)
        step_sim(0.5)

    elif action == "tighten" or action == "fix":
        print(f"  -> Tightening {part}...")
        recolor(parts[part], (0, 0, 1, 1)) # Blue
        show_working_animation(robot, ee_link, parts, part, original_positions=original_positions)
        recolor(parts[part], (0.6, 0.4, 0.2, 1)) # Restore
        step_sim(0.5)

    else:
        # Generic action
        print(f"  -> Processing {part} ({action})...")
        if part in parts:
            safe_call(show_working_animation, robot, ee_link, parts, part, original_positions=original_positions, timeout=12)
        step_sim(0.5)