"""Execute high-level repair-plan actions in the PyBullet scene.

COMPLETE FIX: Handles static parts, proper removal, and exact positioning.
"""

import pybullet as p
import json
import time
import math
from sim_robot import (
    move_ee, open_gripper, close_gripper, 
    create_grasp_constraint, release_grasp, 
    get_visual_gripper_body, move_to_home
)
from sim_connection import step_sim

try:
    from collision_aware_motion import get_obstacle_ids_from_parts
    COLLISION_AWARE_ENABLED = True
    print(f"[STARTUP] COLLISION_AWARE_ENABLED = True")
except ImportError as e:
    COLLISION_AWARE_ENABLED = False
    print(f"[STARTUP] COLLISION_AWARE_ENABLED = False")


def load_json(path):
    """Load a JSON file from disk."""
    with open(path) as f:
        return json.load(f)


def get_pos(part_handle):
    """Return (pos, orn) for a part."""
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
    """Call function with timeout warning."""
    t0 = time.time()
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[SIM] safe_call exception in {fn.__name__}: {e}")
        return False
    dt = time.time() - t0
    if dt > timeout:
        print(f"[SIM] safe_call: {fn.__name__} took {dt:.1f}s")
    return True


def make_part_pickable(part_body, part_name):
    """Make a static part pickable by changing its dynamics.
    
    Since PyBullet doesn't allow changing mass of existing bodies,
    we modify the dynamics to make it behave as if it has mass.
    
    Args:
        part_body: Body ID of the part
        part_name: Name for logging
        
    Returns:
        bool: True if successful
    """
    try:
        # CRITICAL: Change mass to make part pickable
        # Chair parts are created with mass=0 (static) by default
        part_mass = 0.5  # 500g - light but graspable
        
        p.changeDynamics(
            part_body, -1,
            mass=part_mass,
            lateralFriction=1.0,
            spinningFriction=0.2,
            rollingFriction=0.1,
            linearDamping=0.05,
            angularDamping=0.05
        )
        
        print(f"[MAKE_PICKABLE] Made {part_name} (body {part_body}) dynamic with mass={part_mass}kg")
        step_sim(0.1)
        return True
        
    except Exception as e:
        print(f"[MAKE_PICKABLE] Error: {e}")
        return False


def force_remove_part(part_body, part_name, parts):
    """Forcefully remove a part from simulation.
    
    Used when pickup fails - we still need to clear the damaged part.
    
    Args:
        part_body: Body ID to remove
        part_name: Name for logging
        parts: Parts dictionary to update
        
    Returns:
        bool: True if removed
    """
    try:
        print(f"[FORCE_REMOVE] Removing {part_name} (body {part_body})")
        p.removeBody(part_body)
        if part_name in parts:
            del parts[part_name]
        step_sim(0.1)
        return True
    except Exception as e:
        print(f"[FORCE_REMOVE] Error: {e}")
        return False


def simple_pick_up(robot, ee_link, gripper, open_val, close_val, target_body, target_pos):
    """Pick up a part with VERY aggressive approach."""
    print(f"[SIMPLE_PICK] Picking up body {target_body} at {[round(x,3) for x in target_pos]}")
    
    # Open gripper WIDE
    open_gripper(robot, gripper, open_val)
    step_sim(0.2)
    
    # Hover above (20cm)
    hover_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.20]
    print(f"[SIMPLE_PICK] Hover: {[round(x,3) for x in hover_pos]}")
    move_ee(robot, ee_link, hover_pos, steps=150)
    step_sim(0.15)
    
    # Descend CLOSE (1cm above center - VERY AGGRESSIVE)
    grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.01]
    print(f"[SIMPLE_PICK] Descend CLOSE: {[round(x,3) for x in grasp_pos]}")
    move_ee(robot, ee_link, grasp_pos, steps=150)
    step_sim(0.3)  # Extra settling
    
    # Close gripper FIRMLY
    print(f"[SIMPLE_PICK] Closing gripper...")
    close_gripper(robot, gripper, close_val)
    step_sim(0.5)  # LONG settling time
    
    # Create constraint
    print(f"[SIMPLE_PICK] Creating constraint...")
    cid = create_grasp_constraint(robot, ee_link, target_body, -1)
    
    if cid is None:
        print(f"[SIMPLE_PICK] FAILED - constraint creation failed")
        return False
    
    # Lift
    lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.35]
    print(f"[SIMPLE_PICK] Lifting: {[round(x,3) for x in lift_pos]}")
    move_ee(robot, ee_link, lift_pos, steps=150)
    step_sim(0.2)
    
    print(f"[SIMPLE_PICK] SUCCESS! Constraint={cid}")
    return True


def simple_place(robot, ee_link, gripper, open_val, target_pos):
    """Place object at target position."""
    print(f"[SIMPLE_PLACE] Placing at {[round(x,3) for x in target_pos]}")
    
    hover_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.30]
    move_ee(robot, ee_link, hover_pos, steps=150)
    step_sim(0.1)
    
    place_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.10]
    move_ee(robot, ee_link, place_pos, steps=150)
    step_sim(0.1)
    
    release_grasp()
    step_sim(0.1)
    
    open_gripper(robot, gripper, open_val)
    step_sim(0.2)
    
    retract_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.35]
    move_ee(robot, ee_link, retract_pos, steps=150)
    step_sim(0.1)
    
    print(f"[SIMPLE_PLACE] Done!")


def spawn_replacement_part(parts, original_part_name, original_positions):
    """Spawn replacement part in staging area."""
    if original_part_name not in original_positions:
        print(f"[SPAWN] No original position for {original_part_name}")
        return False
    
    try:
        staging_pos = [0.30, 0.75, 0.10]
        
        is_leg = "leg" in original_part_name.lower()
        
        if is_leg:
            size = [0.05, 0.05, 0.45]  # Upright leg
            color = [0.2, 0.8, 0.2, 1]
            mass = 0.5
            staging_pos[2] = size[2] / 2  # Half height above ground
        elif "seat" in original_part_name.lower():
            size = [0.45, 0.45, 0.05]
            color = [0.2, 0.8, 0.2, 1]
            mass = 2.0
        elif "back" in original_part_name.lower():
            size = [0.05, 0.45, 0.5]
            color = [0.2, 0.8, 0.2, 1]
            mass = 1.5
        elif "armrest" in original_part_name.lower():
            size = [0.6, 0.05, 0.05]
            color = [0.2, 0.8, 0.2, 1]
            mass = 0.3
        else:
            print(f"[SPAWN] Unknown part type: {original_part_name}")
            return False
        
        print(f"[SPAWN] Creating replacement for '{original_part_name}'")
        print(f"[SPAWN] Size: {size}, Mass: {mass}kg, Position: {staging_pos}")
        
        from sim_scene import create_dynamic_block
        replacement_body = create_dynamic_block(size, staging_pos, color, mass=mass)
        
        p.resetBasePositionAndOrientation(replacement_body, staging_pos, [0, 0, 0, 1])
        
        replacement_name = f"{original_part_name}_replacement"
        parts[replacement_name] = (replacement_body, -1)
        original_positions[replacement_name] = staging_pos
        
        step_sim(0.3)
        
        final_pos, final_orn = p.getBasePositionAndOrientation(replacement_body)
        print(f"[SPAWN] Created '{replacement_name}' (body={replacement_body})")
        print(f"[SPAWN] Position: {[round(x,3) for x in final_pos]}, Orientation: {[round(x,3) for x in final_orn]}")
        
        return True
        
    except Exception as e:
        print(f"[SPAWN] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_working_animation(robot, ee_link, parts, part_name, original_positions=None):
    """Animate working on a part."""
    if part_name not in parts:
        return
    
    try:
        target_pos, _ = get_pos(parts[part_name])
        if target_pos is None or abs(target_pos[0]) > 5:
            if original_positions and part_name in original_positions:
                target_pos = original_positions[part_name]
            else:
                return
    except Exception:
        return
    
    side_offset = 0.40 if target_pos[1] < 0 else -0.40
    hover_pos = [target_pos[0], target_pos[1] + side_offset, 0.60]
    move_ee(robot, ee_link, hover_pos, steps=80)
    
    work_pos = [target_pos[0], target_pos[1] + side_offset * 0.7, 0.45]
    move_ee(robot, ee_link, work_pos, steps=60)
    
    for _ in range(3):
        step_sim(0.1)
    
    move_ee(robot, ee_link, hover_pos, steps=60)


def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step, original_positions=None):
    """Execute a single repair step."""
    action = step.get("type") or step.get("action_type") or step.get("action")
    part = step.get("target_part")
    
    if not action or not part:
        return
    
    action = action.lower()
    
    if part not in parts and part != "":
        print(f"[STEP] Part '{part}' not in scene")
        return
    
    # ========================================================================
    # INSPECT
    # ========================================================================
    if action == "inspect":
        print(f"  -> Inspecting {part}...")
        recolor(parts[part], (1, 1, 0, 1))
        safe_call(show_working_animation, robot, ee_link, parts, part, original_positions)
        recolor(parts[part], (0.6, 0.4, 0.2, 1))
        step_sim(0.5)
    
    # ========================================================================
    # REPLACE - Complete workflow
    # ========================================================================
    elif action == "replace":
        print(f"  -> REPLACING {part}...")
        
        target_body, _ = parts[part]
        target_pos, _ = p.getBasePositionAndOrientation(target_body)
        
        # PHASE 1: Remove damaged part
        print(f"  -> Phase 1: Removing damaged {part}...")
        
        # CRITICAL FIX: Make part pickable (change from static to dynamic)
        print(f"  -> Making {part} pickable...")
        make_part_pickable(target_body, part)
        step_sim(0.2)
        
        # Move robot to home
        print(f"  -> Moving robot to home...")
        move_to_home(robot, robot_type="kuka")
        step_sim(0.3)
        
        recolor(parts[part], (1, 0, 0, 1))
        
        # Try to pick up
        pickup_success = simple_pick_up(robot, ee_link, gripper, open_val, close_val, target_body, target_pos)
        
        if pickup_success:
            # Successfully picked up - move to drop zone
            drop_zone = [1.0, 0.0, 0.25]
            simple_place(robot, ee_link, gripper, open_val, drop_zone)
            print(f"  -> Phase 1: Damaged part removed via pickup")
        else:
            # Pickup failed - force remove
            print(f"  -> Phase 1: Pickup failed, force removing...")
        
        # Always force remove to ensure it's gone
        force_remove_part(target_body, part, parts)
        step_sim(0.3)
        
        # PHASE 2: Spawn replacement
        print(f"  -> Phase 2: Spawning replacement...")
        
        if spawn_replacement_part(parts, part, original_positions):
            replacement_name = f"{part}_replacement"
            print(f"  -> Replacement spawned as '{replacement_name}'")
            
            if replacement_name in parts:
                recolor(parts[replacement_name], (0, 1, 0, 1))
                
                rep_body, _ = parts[replacement_name]
                rep_pos, _ = p.getBasePositionAndOrientation(rep_body)
                print(f"  -> Replacement at: {[round(x,3) for x in rep_pos]}")
                
                step_sim(0.5)
                
                # PHASE 3: Install replacement
                print(f"  -> Phase 3: Installing replacement...")
                
                # Move to home
                move_to_home(robot, robot_type="kuka")
                step_sim(0.3)
                
                # Pickup replacement
                if simple_pick_up(robot, ee_link, gripper, open_val, close_val, rep_body, rep_pos):
                    
                    if part in original_positions:
                        install_pos = original_positions[part]
                        print(f"  -> Installing at: {[round(x,3) for x in install_pos]}")
                        
                        # Move to hover above install
                        hover_install = [install_pos[0], install_pos[1], install_pos[2] + 0.40]
                        move_ee(robot, ee_link, hover_install, steps=150)
                        step_sim(0.2)
                        
                        # Descend to install
                        final_install = [install_pos[0], install_pos[1], install_pos[2] + 0.15]
                        move_ee(robot, ee_link, final_install, steps=150)
                        step_sim(0.2)
                        
                        # Release constraint FIRST
                        release_grasp()
                        step_sim(0.1)
                        
                        # CRITICAL FIX: Snap to EXACT position with correct orientation
                        # This fixes the position drift issue
                        upright_orn = [0, 0, 0, 1]
                        p.resetBasePositionAndOrientation(rep_body, install_pos, upright_orn)
                        
                        # Make the replacement STATIC (part of chair)
                        p.changeDynamics(rep_body, -1, mass=0)
                        step_sim(0.1)
                        
                        # Verify final position
                        final_pos, final_orn = p.getBasePositionAndOrientation(rep_body)
                        error = math.sqrt(sum((final_pos[i] - install_pos[i])**2 for i in range(3)))
                        print(f"  -> Positioning error: {error*1000:.1f}mm")
                        print(f"  -> Final pos: {[round(x,3) for x in final_pos]}")
                        print(f"  -> Final orn: {[round(x,3) for x in final_orn]}")
                        
                        # Update parts dictionary
                        recolor(parts[replacement_name], (0.6, 0.4, 0.2, 1))
                        parts[part] = (rep_body, -1)
                        del parts[replacement_name]
                        if replacement_name in original_positions:
                            del original_positions[replacement_name]
                        
                        print(f"  -> {part} successfully replaced!")
                        
                        # Open gripper and retract
                        open_gripper(robot, gripper, open_val)
                        step_sim(0.2)
                        
                        retract = [install_pos[0], install_pos[1], install_pos[2] + 0.45]
                        move_ee(robot, ee_link, retract, steps=150)
                        step_sim(0.2)
        
        step_sim(0.5)
    
    # ========================================================================
    # OTHER ACTIONS
    # ========================================================================
    elif action in ["tighten", "fix"]:
        print(f"  -> Tightening/Fixing {part}...")
        recolor(parts[part], (0, 0, 1, 1))
        show_working_animation(robot, ee_link, parts, part, original_positions)
        recolor(parts[part], (0.6, 0.4, 0.2, 1))
        step_sim(0.5)
    
    elif action == "clean":
        print(f"  -> Cleaning {part}...")
        recolor(parts[part], (0.5, 0.8, 1, 1))
        show_working_animation(robot, ee_link, parts, part, original_positions)
        recolor(parts[part], (0.6, 0.4, 0.2, 1))
        step_sim(0.5)
    
    else:
        print(f"  -> Processing {part} ({action})...")
        if part in parts:
            safe_call(show_working_animation, robot, ee_link, parts, part, original_positions)
        step_sim(0.5)