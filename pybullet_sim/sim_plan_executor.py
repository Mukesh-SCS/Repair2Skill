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


def pick_up_part(robot, ee_link, gripper, open_val, close_val, parts, part_name, original_positions=None):
    """Pick up a part using the gripper.
    
    This function:
    1. Opens the gripper
    2. Moves the gripper above the part
    3. Moves down to the part
    4. Closes the gripper (or simulates by moving close)
    5. Lifts the part up
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        gripper: List of gripper joint indices (can be empty)
        open_val: Gripper open value
        close_val: Gripper close value
        parts: Dictionary of part names to (body_id, link_index)
        part_name: Name of the part to pick up
        original_positions: Dict of original part positions for reference
    """
    if part_name not in parts:
        print(f"[PICKUP] Part '{part_name}' not in scene")
        return False
    
    try:
        target_pos, _ = get_pos(parts[part_name])
        
        # Validate position
        invalid = False
        if target_pos is None:
            invalid = True
        else:
            tx, ty, tz = target_pos
            if abs(tx) > 5.0 or abs(ty) > 5.0 or tz < -1.0:
                invalid = True
        
        if invalid and original_positions and part_name in original_positions:
            stored = original_positions[part_name]
            print(f"[PICKUP] Part '{part_name}' appears invalid; using original pos {stored}")
            target_pos = stored
        
        if target_pos is None:
            print(f"[PICKUP] Could not get position for {part_name}")
            return False
            
    except Exception as e:
        print(f"[PICKUP] Could not get position of {part_name}: {e}")
        return False
    
    # Step 1: Open gripper (if it exists)
    if gripper and len(gripper) > 0:
        print(f"[PICKUP] Opening gripper...")
        open_gripper(robot, gripper, open_val)
        step_sim(0.1)
    else:
        print(f"[PICKUP] No gripper joints available, simulating pickup...")
    
    # Step 2: Move to hover position (above the part)
    try:
        hover_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.25]
        print(f"[PICKUP] Moving to hover position: {hover_pos}")
        move_ee(robot, ee_link, hover_pos, steps=80)
    except Exception as e:
        print(f"[PICKUP] Could not move to hover: {e}")
        step_sim(0.2)
        return False
    
    # Step 3: Move down to grasp position
    try:
        grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.08]
        print(f"[PICKUP] Moving to grasp position: {grasp_pos}")
        move_ee(robot, ee_link, grasp_pos, steps=60)
    except Exception as e:
        print(f"[PICKUP] Could not move to grasp: {e}")
        step_sim(0.2)
        return False
    
    # Step 4: Close gripper (or simulate with brief pause)
    if gripper and len(gripper) > 0:
        print(f"[PICKUP] Closing gripper to grasp {part_name}...")
        close_gripper(robot, gripper, close_val)
    else:
        print(f"[PICKUP] Simulating grasp of {part_name}...")
        step_sim(0.2)
    
    step_sim(0.3)
    
    # Step 5: Lift the part
    try:
        lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.35]
        print(f"[PICKUP] Lifting {part_name}...")
        move_ee(robot, ee_link, lift_pos, steps=60)
    except Exception as e:
        print(f"[PICKUP] Could not lift part: {e}")
        step_sim(0.2)
        return False
    
    step_sim(0.3)
    return True


def place_part(robot, ee_link, gripper, open_val, close_val, parts, part_name, drop_zone=None):
    """Place a part at a drop zone location.
    
    This function:
    1. Moves to drop zone
    2. Opens gripper
    3. Returns to neutral position
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        gripper: List of gripper joint indices (can be empty)
        open_val: Gripper open value
        close_val: Gripper close value
        parts: Dictionary of parts (not used but kept for consistency)
        part_name: Name of part being placed (for logging)
        drop_zone: Target position [x, y, z]. If None, uses default.
    """
    if drop_zone is None:
        drop_zone = [1.0, 0.0, 0.3]  # Default drop zone (to the side)
    
    try:
        print(f"[PLACE] Moving {part_name} to drop zone: {drop_zone}")
        move_ee(robot, ee_link, drop_zone, steps=80)
    except Exception as e:
        print(f"[PLACE] Could not move to drop zone: {e}")
        step_sim(0.2)
        return False
    
    step_sim(0.2)
    
    # Open gripper to release part
    if gripper and len(gripper) > 0:
        print(f"[PLACE] Opening gripper to release {part_name}...")
        open_gripper(robot, gripper, open_val)
    else:
        print(f"[PLACE] Releasing {part_name}...")
        step_sim(0.1)
    
    step_sim(0.3)
    return True


def spawn_replacement_part(parts, original_part_name, original_positions, spawn_offset=[0.2, 0.2, 0.3]):
    """Spawn a new replacement part near the original position.
    
    This creates a new dynamic (movable) version of a part that can be picked up.
    
    Args:
        parts: Dictionary of parts to add the new part to
        original_part_name: Name of the part being replaced
        original_positions: Dict of original positions
        spawn_offset: Offset from original position where replacement spawns [x, y, z]
    
    Returns:
        bool: True if replacement was successfully spawned
    """
    if original_part_name not in original_positions:
        print(f"[SPAWN] No original position for {original_part_name}")
        return False
    
    try:
        original_pos = original_positions[original_part_name]
        
        # Determine part dimensions based on name
        if "leg" in original_part_name:
            size = [0.05, 0.05, 0.45]
            color = [0.6, 0.4, 0.2, 1]  # Brown (new/undamaged)
        elif original_part_name == "seat":
            size = [0.45, 0.45, 0.05]
            color = [0.6, 0.4, 0.2, 1]
        elif original_part_name == "back":
            size = [0.05, 0.45, 0.5]
            color = [0.6, 0.4, 0.2, 1]
        elif "armrest" in original_part_name:
            size = [0.6, 0.05, 0.05]
            color = [0.6, 0.4, 0.2, 1]
        else:
            print(f"[SPAWN] Unknown part type: {original_part_name}")
            return False
        
        # Calculate spawn position (offset from original)
        spawn_pos = [
            original_pos[0] + spawn_offset[0],
            original_pos[1] + spawn_offset[1],
            original_pos[2] + spawn_offset[2]
        ]
        
        # Create a dynamic replacement part
        from sim_scene import block
        replacement_body = block(size, spawn_pos, color)
        
        # Add to parts dict with a special name
        replacement_name = f"{original_part_name}_replacement"
        parts[replacement_name] = (replacement_body, -1)
        
        print(f"[SPAWN] Created replacement for '{original_part_name}' at {spawn_pos}")
        return True
        
    except Exception as e:
        print(f"[SPAWN] Error spawning replacement: {e}")
        return False 


def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step, original_positions=None):
    """Execute a single repair step.
    
    Enhanced to support gripper-based part removal and replacement:
    - "remove": Pick up the damaged part with gripper and place in drop zone
    - "replace": Spawn a new replacement part and pick it up
    - "attach": Pick up the replacement and move it back to original position
    - "inspect", "tighten", Show working animation
    """
    
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
        recolor(parts[part], (0.6, 0.4, 0.2, 1)) # Restore brown
        step_sim(0.5)

    elif action == "remove":
        print(f"  -> REMOVING {part} with gripper...")
        # Highlight the part being removed
        recolor(parts[part], (1, 0, 0, 1)) # Red (damaged)
        step_sim(0.2)
        
        # Pick up the damaged part with gripper
        if pick_up_part(robot, ee_link, gripper, open_val, close_val, parts, part, original_positions=original_positions):
            # Move it to drop zone
            drop_zone = [1.2, 0.0, 0.3]
            place_part(robot, ee_link, gripper, open_val, close_val, parts, part, drop_zone=drop_zone)
            
            # Move part far away (remove from scene)
            body, link = parts[part]
            p.resetBasePositionAndOrientation(body, [10, 10, -10], [0, 0, 0, 1])
            print(f"  -> {part} removed and set aside")
        
        step_sim(0.5)

    elif action == "replace":
        print(f"  -> REPLACING {part} (removing damaged + installing new)...")
        
        # Step 1: Remove the damaged part first
        recolor(parts[part], (1, 0, 0, 1))  # Red (damaged)
        step_sim(0.2)
        
        # Pick up the damaged part with gripper
        if pick_up_part(robot, ee_link, gripper, open_val, close_val, parts, part, original_positions=original_positions):
            # Move it to drop zone
            drop_zone = [1.2, 0.0, 0.3]
            place_part(robot, ee_link, gripper, open_val, close_val, parts, part, drop_zone=drop_zone)
            
            # Move part far away (remove from scene)
            body, link = parts[part]
            p.resetBasePositionAndOrientation(body, [10, 10, -10], [0, 0, 0, 1])
            print(f"  -> {part} removed")
        
        step_sim(0.3)
        
        # Step 2: Spawn a new replacement part
        if spawn_replacement_part(parts, part, original_positions, spawn_offset=[0.3, 0.0, 0.15]):
            replacement_name = f"{part}_replacement"
            print(f"  -> Replacement spawned as '{replacement_name}'")
            
            # Highlight the replacement in green
            if replacement_name in parts:
                recolor(parts[replacement_name], (0, 1, 0, 1))  # Green (new/good)
            
            step_sim(0.3)
            
            # Step 3: Install the replacement part
            print(f"  -> Installing replacement...")
            if pick_up_part(robot, ee_link, gripper, open_val, close_val, parts, replacement_name, original_positions=original_positions):
                # Move it back to the original position of the damaged part
                if original_positions and part in original_positions:
                    original_pos = original_positions[part]
                    # Move to installation position (slightly above)
                    install_pos = [original_pos[0], original_pos[1], original_pos[2] + 0.15]
                    
                    try:
                        print(f"  -> Moving to installation position: {install_pos}")
                        move_ee(robot, ee_link, install_pos, steps=80)
                    except Exception as e:
                        print(f"  -> Could not move to installation position: {e}")
                    
                    step_sim(0.2)
                    
                    # Open gripper to release the replacement
                    if gripper and len(gripper) > 0:
                        print(f"  -> Releasing replacement part")
                        open_gripper(robot, gripper, open_val)
                    else:
                        print(f"  -> Releasing replacement part (simulated)")
                        step_sim(0.1)
                    
                    step_sim(0.2)
                    
                    # Update the part's visual position to the original location
                    try:
                        body, link = parts[part]
                        p.resetBasePositionAndOrientation(body, original_pos, [0, 0, 0, 1])
                        # Change color back to normal (undamaged)
                        recolor(parts[part], (0.6, 0.4, 0.2, 1))
                        print(f"  -> {part} successfully replaced and installed!")
                    except Exception as e:
                        print(f"  -> Error repositioning part: {e}")
                    
                    step_sim(0.2)
                
                # Remove the replacement part from scene
                try:
                    body, link = parts[replacement_name]
                    p.resetBasePositionAndOrientation(body, [10, 10, -10], [0, 0, 0, 1])
                except Exception:
                    pass
            else:
                print(f"  -> Could not pick up replacement")
        else:
            print(f"  -> Failed to spawn replacement for {part}")
        
        step_sim(0.5)

    elif action == "attach" or action == "install":
        print(f"  -> INSTALLING {part}...")
        replacement_name = f"{part}_replacement"
        
        # If we have a replacement, pick it up and place it
        if replacement_name in parts:
            # Pick up the replacement
            if pick_up_part(robot, ee_link, gripper, open_val, close_val, parts, replacement_name, original_positions=original_positions):
                # Move it back to the original position of the damaged part
                if original_positions and part in original_positions:
                    original_pos = original_positions[part]
                    # Move to installation position (slightly above)
                    install_pos = [original_pos[0], original_pos[1], original_pos[2] + 0.15]
                    
                    try:
                        print(f"  -> Moving replacement to installation position: {install_pos}")
                        move_ee(robot, ee_link, install_pos, steps=80)
                    except Exception as e:
                        print(f"  -> Could not move to installation position: {e}")
                    
                    step_sim(0.2)
                    
                    # Open gripper to release the replacement
                    if gripper and len(gripper) > 0:
                        print(f"  -> Releasing replacement part")
                        open_gripper(robot, gripper, open_val)
                    else:
                        print(f"  -> Releasing replacement part (simulated)")
                        step_sim(0.1)
                    
                    step_sim(0.2)
                    
                    # Update the part's visual position to the original location
                    try:
                        body, link = parts[part]
                        p.resetBasePositionAndOrientation(body, original_pos, [0, 0, 0, 1])
                        # Change color back to normal (undamaged)
                        recolor(parts[part], (0.6, 0.4, 0.2, 1))
                        print(f"  -> {part} successfully replaced and installed!")
                    except Exception as e:
                        print(f"  -> Error repositioning part: {e}")
                    
                    step_sim(0.2)
                
                # Remove the replacement part from scene
                try:
                    body, link = parts[replacement_name]
                    p.resetBasePositionAndOrientation(body, [10, 10, -10], [0, 0, 0, 1])
                except Exception:
                    pass
            else:
                print(f"  -> Could not pick up replacement")
        else:
            # Fallback: just show working animation if no replacement
            print(f"  -> No replacement found, showing working animation")
            if part in parts:
                recolor(parts[part], (0, 1, 0, 1))  # Green
                safe_call(show_working_animation, robot, ee_link, parts, part, original_positions=original_positions, timeout=12)
                recolor(parts[part], (0.6, 0.4, 0.2, 1))
        
        step_sim(0.5)

    elif action == "tighten" or action == "fix":
        print(f"  -> Tightening/Fixing {part}...")
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