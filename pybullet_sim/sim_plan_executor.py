"""Execute high-level repair-plan actions in the PyBullet scene.

This module translates plan steps (strings) into simple robot movements and
visual feedback. It handles both URDF-based parts (links) and procedural
parts (bodies) correctly.

GRASPING IMPLEMENTATION:
------------------------
Uses constraint-based grasping from sim_robot module. When picking up a part:
1. Robot moves above the part (hover position)
2. Gripper opens (if physical gripper exists)
3. Robot moves down to grasp position
4. Constraint is created to "weld" object to end-effector
5. Gripper closes (visual feedback)
6. Robot lifts the part (object moves with EE due to constraint)

When releasing:
1. Robot moves to target position
2. Constraint is removed (object becomes free)
3. Gripper opens
4. Robot retracts

This approach ensures reliable object manipulation regardless of friction settings.
"""

import pybullet as p
import json
import time
import math
from sim_robot import (
    move_ee, open_gripper, close_gripper, 
    create_grasp_constraint, release_grasp, get_grasp_constraint,
    get_visual_gripper_body
)
from sim_connection import step_sim
try:
    from collision_aware_motion import (
        move_ee_collision_safe,
        retreat_collision_safe,
        get_obstacle_ids_from_parts,
        get_collision_filter_manager,
        log_collision_status,
        is_configuration_collision_free,
        check_collision_robot_vs_obstacles
    )
    COLLISION_AWARE_ENABLED = True
except ImportError:
    COLLISION_AWARE_ENABLED = False
    print("[WARNING] collision_aware_motion not available - using direct motion")
import time

# ============================================================================
# GRASP EXECUTION FLAGS
# ============================================================================
# Controls whether angled grasps (front, side, back) are actually executed.
# When True, uses orientation-constrained IK for angled approaches.
# When False, falls back to overhead grasps for all parts.
EXECUTE_ANGLED_GRASPS = True

# Only enable angled grasps for these part categories (safe rollout).
# Set to True to enable angled grasps for that category.
ANGLED_GRASP_ALLOWLIST = {
    "back": True,       # backrest - front approach works well
    "leg": False,       # legs - keep False until verified
    "armrest": False,   # armrests - keep False until verified
    "seat": False,      # seat - too large, keep overhead
}


def part_category(name: str) -> str:
    """Determine part category from name for grasp strategy selection."""
    n = name.lower()
    if "back" in n: return "back"
    if "leg" in n: return "leg"
    if "armrest" in n or "arm" in n: return "armrest"
    if "seat" in n: return "seat"
    return "other"


def load_json(path):
    """Load a JSON file from disk."""
    with open(path) as f:
        return json.load(f)


# ============================================================================
# COLLISION-SAFE MOTION WRAPPER
# ============================================================================

def move_ee_pose(robot, ee_link, pos, orn=None, steps=80):
    """Move EE to a pose (position + optional orientation).
    
    When orientation is provided, uses IK with orientation constraint.
    This enables angled grasps where the gripper approaches from front/side.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        pos: Target position [x, y, z]
        orn: Target orientation quaternion [x, y, z, w] or None for default
        steps: Number of simulation steps
        
    Returns:
        bool: True if motion executed (success not guaranteed for IK)
    """
    if orn is None:
        move_ee(robot, ee_link, pos, steps=steps)
        return True
    
    # IK with orientation constraint
    joints = p.calculateInverseKinematics(
        robot, ee_link, pos, targetOrientation=orn
    )
    num_joints = p.getNumJoints(robot)
    
    # Apply only first 7 arm joints
    for j in range(min(7, num_joints)):
        p.setJointMotorControl2(
            robot, j, p.POSITION_CONTROL,
            targetPosition=joints[j],
            force=500
        )
    
    step_sim(steps / 240.0)
    return True


def score_side_clearance(robot, ee_link, base_pos, parts, excluded_body=None, 
                         side_axis="y", sign=1, probe=0.18):
    """Score clearance on a side for side-approach selection.
    
    Lower score = better (more clearance from obstacles).
    Checks closest points between gripper (or robot) and obstacles.
    Uses gripper body when available since that's what actually collides.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index  
        base_pos: Base position to probe from [x, y, z]
        parts: Dictionary of parts for obstacle detection
        excluded_body: Body ID to exclude (e.g., target part)
        side_axis: 'y' or 'x' - which axis to probe along
        sign: +1 or -1 - which direction along axis
        probe: Distance to probe (default 18cm)
        
    Returns:
        float: Score (lower = more clearance)
    """
    axis_index = 1 if side_axis == "y" else 0
    
    # Compute the probe position: shifted along the approach axis
    probe_pos = list(base_pos)
    probe_pos[axis_index] += sign * probe
    
    obstacle_ids = get_obstacle_ids_from_parts(parts) if COLLISION_AWARE_ENABLED else []
    if excluded_body is not None:
        obstacle_ids = [o for o in obstacle_ids if o != excluded_body]
    
    # Use gripper body if available (it's what actually collides during approach)
    gripper_body = get_visual_gripper_body()
    check_body = gripper_body if gripper_body is not None else robot
    
    # Score based on closest geometry between gripper/robot and obstacles
    score = 0.0
    check_radius = 0.25  # 25cm radius check
    
    for obs in obstacle_ids:
        try:
            # Check closest points between gripper/robot and obstacle
            pts = p.getClosestPoints(bodyA=check_body, bodyB=obs, distance=check_radius)
            if pts:
                # Use minimum distance from all contact pairs
                min_dist = min(pt[8] for pt in pts)  # pt[8] is contact distance
                # Penalize closeness: closer = higher score
                if min_dist < check_radius:
                    score += (check_radius - min_dist)
        except:
            pass
    
    return score


def move_ee_safe(robot, ee_link, pos, parts=None, excluded_part=None, 
                 use_collision_check=True, steps=100):
    """Move end-effector with optional collision checking.
    
    This wrapper uses collision-aware motion when available, falling back
    to direct motion if collision checking is disabled or unavailable.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        pos: Target position [x, y, z]
        parts: Dictionary of parts (for obstacle detection)
        excluded_part: Part name to exclude from collision checking (e.g., grasped part)
        use_collision_check: Whether to use collision-aware motion
        steps: Physics steps for motion
        
    Returns:
        bool: True if motion succeeded, False if blocked by collision
    """
    if not use_collision_check or not COLLISION_AWARE_ENABLED or parts is None:
        # Direct motion (no collision checking)
        try:
            move_ee(robot, ee_link, pos, steps=steps)
            return True
        except Exception as e:
            print(f"[MOTION] Direct motion failed: {e}")
            return False
    
    # Collision-aware motion
    try:
        obstacle_ids = get_obstacle_ids_from_parts(parts)
        
        # Build excluded obstacles set
        excluded = set()
        if excluded_part and excluded_part in parts:
            part_body, _ = parts[excluded_part]
            excluded.add(part_body)
        
        success, message = move_ee_collision_safe(
            robot, ee_link, pos, obstacle_ids,
            excluded_obstacles=excluded,
            use_pre_approach=True,
            use_rrt_fallback=True,
            steps=steps
        )
        
        if not success:
            print(f"[MOTION] Collision-safe motion failed: {message}")
            # DO NOT fall back to direct motion - that defeats the purpose
            # Return False so caller can handle the failure appropriately
            return False
        
        return True
        
    except Exception as e:
        print(f"[MOTION] Collision-aware motion error: {e}")
        # Return False on error - don't blindly execute potentially colliding motion
        return False


def check_motion_collision(robot, ee_link, target_pos, parts, excluded_part=None):
    """Check if a motion would result in collision without executing it.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        target_pos: Target position to check
        parts: Dictionary of parts
        excluded_part: Part to exclude from checking
        
    Returns:
        Tuple of (would_collide, reason)
    """
    if not COLLISION_AWARE_ENABLED:
        return False, "Collision checking not available"
    
    try:
        obstacle_ids = get_obstacle_ids_from_parts(parts)
        excluded = set()
        if excluded_part and excluded_part in parts:
            part_body, _ = parts[excluded_part]
            excluded.add(part_body)
        
        # Get IK solution
        target_joints = p.calculateInverseKinematics(robot, ee_link, target_pos)
        target_joints = list(target_joints)[:7]  # Arm joints only
        
        is_free, reason = is_configuration_collision_free(
            robot, target_joints, obstacle_ids, excluded_obstacles=excluded
        )
        
        return not is_free, reason
        
    except Exception as e:
        return False, f"Check failed: {e}"


# =============================================================================
# SEMANTIC GRASP POSE SELECTION
# =============================================================================
# Chooses grasp approach direction based on part type and geometry.
# This avoids "straight down into narrow gaps" collisions and produces
# more believable grasps (backrest from front, legs from side, etc.)
# =============================================================================

def get_part_dimensions(body_id):
    """Get the dimensions of a part from its visual shape data.
    
    Returns:
        Tuple of (width, depth, height) or None if unable to determine
    """
    try:
        visual_data = p.getVisualShapeData(body_id)
        if visual_data and len(visual_data) > 0:
            shape_type = visual_data[0][2]  # GEOM_BOX = 3
            if shape_type == p.GEOM_BOX:
                half_extents = visual_data[0][3]  # (hx, hy, hz)
                return [2 * h for h in half_extents]  # Full dimensions
        
        # Fallback: use AABB
        aabb_min, aabb_max = p.getAABB(body_id)
        return [aabb_max[i] - aabb_min[i] for i in range(3)]
    except:
        return None


def select_grasp_strategy(part_name, body_id):
    """Select grasp approach direction based on part type and geometry.
    
    Returns a dictionary with:
        approach_dir: 'above', 'front', 'side', 'back'
        approach_offset: [x, y, z] offset from part center for pre-approach
        grasp_offset: [x, y, z] offset for final grasp position
        ee_orientation: Optional EE orientation quaternion for grasp
        gripper_open: Suggested gripper open value (wider for bigger parts)
        
    Strategy logic:
        - Legs/armrests (long thin): grasp from side (±Y or ±X approach)
        - Backrest: grasp from front (±X approach)  
        - Seat: grasp from side (if at all - usually too big)
        - Default: from above (overhead grasp)
    """
    import math
    
    dims = get_part_dimensions(body_id)
    if dims is None:
        dims = [0.1, 0.1, 0.1]  # Fallback
    
    width, depth, height = dims
    part_lower = part_name.lower()
    
    # Default strategy (overhead grasp)
    strategy = {
        'approach_dir': 'above',
        'approach_offset': [0, 0, 0.15],   # 15cm above
        'grasp_offset': [0, 0, 0.10],      # 10cm above (TCP at center)
        'ee_orientation': None,             # Default orientation
        'gripper_open': 0.0,                # Standard open
    }
    
    # Legs: tall thin parts - grasp from side
    if 'leg' in part_lower:
        # Legs are typically taller than wide - grasp from side
        # Approach from +Y or -Y depending on position
        strategy['approach_dir'] = 'side'
        strategy['approach_offset'] = [0, 0.12, 0]   # 12cm to side
        strategy['grasp_offset'] = [0, 0.02, 0]      # Close to side
        # Wider opening for leg cross-section
        strategy['gripper_open'] = 0.0
        print(f"[GRASP] Leg detected ({width:.2f}x{depth:.2f}x{height:.2f}m) - side approach")
    
    # Armrests: long horizontal parts - grasp from side/above
    elif 'armrest' in part_lower or 'arm' in part_lower:
        # Armrests are typically long and thin horizontally
        strategy['approach_dir'] = 'above'  # Still from above but at an angle
        strategy['approach_offset'] = [0, 0, 0.12]
        strategy['grasp_offset'] = [0, 0, 0.08]
        print(f"[GRASP] Armrest detected ({width:.2f}x{depth:.2f}x{height:.2f}m) - above approach")
    
    # Backrest/back: tall vertical - grasp from front
    elif 'back' in part_lower:
        # Backrest - approach from front (+X direction, toward robot)
        strategy['approach_dir'] = 'front'
        strategy['approach_offset'] = [0.15, 0, 0]   # 15cm in front
        strategy['grasp_offset'] = [0.05, 0, 0]      # 5cm from front face
        print(f"[GRASP] Backrest detected ({width:.2f}x{depth:.2f}x{height:.2f}m) - front approach")
    
    # Seat: large flat part - generally avoid, but side grasp if needed
    elif 'seat' in part_lower:
        strategy['approach_dir'] = 'side'
        strategy['approach_offset'] = [0, 0.20, 0]   # 20cm to side (seat is big)
        strategy['grasp_offset'] = [0, 0.05, 0]
        strategy['gripper_open'] = 0.0  # May need wider
        print(f"[GRASP] Seat detected ({width:.2f}x{depth:.2f}x{height:.2f}m) - side approach (caution: large)")
    
    else:
        # Default: overhead grasp
        print(f"[GRASP] Generic part ({width:.2f}x{depth:.2f}x{height:.2f}m) - overhead approach")
    
    return strategy


def compute_approach_positions(target_pos, strategy, hover_height=0.25, tcp_offset=0.10, side_sign=1):
    """Compute hover and grasp positions based on grasp strategy.
    
    Args:
        target_pos: Part center position [x, y, z]
        strategy: Grasp strategy dict from select_grasp_strategy()
        hover_height: Additional hover clearance (default 25cm)
        tcp_offset: TCP offset for gripper (default 10cm)
        side_sign: For side approaches, +1 = +Y side, -1 = -Y side
        
    Returns:
        Tuple of (hover_pos, grasp_pos, ee_orn)
    """
    approach_dir = strategy['approach_dir']
    approach_offset = strategy['approach_offset']
    grasp_offset = strategy['grasp_offset']
    
    if approach_dir == 'above':
        # Standard overhead approach
        hover_pos = [
            target_pos[0] + approach_offset[0],
            target_pos[1] + approach_offset[1],
            target_pos[2] + hover_height
        ]
        grasp_pos = [
            target_pos[0] + grasp_offset[0],
            target_pos[1] + grasp_offset[1],
            target_pos[2] + tcp_offset
        ]
        ee_orn = None  # Default orientation (pointing down)
        
    elif approach_dir == 'front':
        # Approach from front (+X), gripper pointing toward -X
        hover_pos = [
            target_pos[0] + approach_offset[0] + 0.10,  # Extra clearance
            target_pos[1],
            target_pos[2]
        ]
        grasp_pos = [
            target_pos[0] + grasp_offset[0],
            target_pos[1],
            target_pos[2]
        ]
        # Rotate gripper to point toward -X (90° pitch)
        ee_orn = p.getQuaternionFromEuler([0, 1.57, 0])  # 90° around Y
        
    elif approach_dir == 'side':
        # Approach from side (+Y or -Y based on side_sign parameter)
        # side_sign = 1 means +Y, side_sign = -1 means -Y
        hover_pos = [
            target_pos[0],
            target_pos[1] + side_sign * (abs(approach_offset[1]) + 0.10),
            target_pos[2]
        ]
        grasp_pos = [
            target_pos[0],
            target_pos[1] + side_sign * abs(grasp_offset[1]),
            target_pos[2]
        ]
        # Rotate gripper to point toward ±Y (90° roll)
        # Adjust roll direction based on side_sign
        roll_angle = 1.57 if side_sign > 0 else -1.57
        ee_orn = p.getQuaternionFromEuler([roll_angle, 0, 0])
        
    elif approach_dir == 'back':
        # Approach from back (-X)
        hover_pos = [
            target_pos[0] - approach_offset[0] - 0.10,
            target_pos[1],
            target_pos[2]
        ]
        grasp_pos = [
            target_pos[0] - grasp_offset[0],
            target_pos[1],
            target_pos[2]
        ]
        ee_orn = p.getQuaternionFromEuler([0, -1.57, 0])  # -90° around Y
        
    else:
        # Fallback to overhead
        hover_pos = [target_pos[0], target_pos[1], target_pos[2] + hover_height]
        grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + tcp_offset]
        ee_orn = None
    
    return hover_pos, grasp_pos, ee_orn


def approach_linear(robot, ee_link, start_pos, end_pos, steps=25, 
                    stop_on_contact=True, contact_bodies=None, gripper_body=None,
                    use_compliance=True):
    """Move the end-effector linearly from start to end, with contact detection.
    
    This provides realistic collision-aware motion without disabling physics.
    The robot will stop if it contacts any of the specified bodies.
    
    COMPLIANCE MODE: When enabled, uses "soft contact" behavior:
    - Reduces motor force when near contact (proximity < 3cm)
    - Uses smaller steps during final approach phase
    - More frequent collision checks near target
    This prevents "bulldozing" and reduces jitter during grasp.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        start_pos: Starting position [x, y, z]
        end_pos: Target position [x, y, z]
        steps: Number of interpolation steps (default 25)
        stop_on_contact: If True, stop when contact detected
        contact_bodies: List of body IDs to check for contact with
        gripper_body: Optional gripper body ID to also check for contacts
        use_compliance: If True, reduce forces near contact (default True)
        
    Returns:
        bool: True if reached end without contact, False if stopped early
    """
    import math
    
    # Compliance parameters
    NEAR_CONTACT_THRESHOLD = 0.03  # 3cm - switch to soft mode
    NORMAL_MOTOR_FORCE = 500
    SOFT_MOTOR_FORCE = 150
    NORMAL_MAX_VEL = 3.0
    SOFT_MAX_VEL = 0.5
    
    total_distance = math.sqrt(sum((end_pos[i] - start_pos[i])**2 for i in range(3)))
    
    for i in range(1, steps + 1):
        t = i / steps
        pos = [
            start_pos[0] + t * (end_pos[0] - start_pos[0]),
            start_pos[1] + t * (end_pos[1] - start_pos[1]),
            start_pos[2] + t * (end_pos[2] - start_pos[2]),
        ]
        
        # Calculate remaining distance to target
        remaining = math.sqrt(sum((end_pos[j] - pos[j])**2 for j in range(3)))
        
        # Compliance: use smaller steps when close to target (near contact zone)
        # This provides softer approach without ineffective motor control changes
        if use_compliance and remaining < NEAR_CONTACT_THRESHOLD:
            # Near-contact mode: finer steps for gentler approach
            move_steps = 5
        else:
            move_steps = 10
        
        move_ee(robot, ee_link, pos, steps=move_steps)
        p.stepSimulation()
        
        if stop_on_contact and contact_bodies:
            # Check for contact between robot and any obstacle body
            for b in contact_bodies:
                contacts = p.getContactPoints(bodyA=robot, bodyB=b)
                if contacts:
                    print(f"[APPROACH] Robot contact detected with body {b} at step {i}/{steps}")
                    return False
                # Also check gripper body contacts (for KUKA visual gripper)
                if gripper_body is not None:
                    gripper_contacts = p.getContactPoints(bodyA=gripper_body, bodyB=b)
                    if gripper_contacts:
                        print(f"[APPROACH] Gripper contact detected with body {b} at step {i}/{steps}")
                        return False
    
    return True


def retreat_mirrored(robot, ee_link, grasp_pos, hover_pos, parts, excluded_part=None, steps=25):
    """Retreat by reversing the final approach: grasp_pos -> hover_pos.
    
    This avoids lateral motion near obstacles by following the same path
    we used to approach, but in reverse. Much safer than arbitrary retreat.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        grasp_pos: Position we grasped from [x, y, z]
        hover_pos: Position we hovered at before approach [x, y, z]
        parts: Dictionary of parts for collision detection
        excluded_part: Part name to exclude from collision checks
        steps: Number of steps for retreat motion
        
    Returns:
        bool: True if retreat succeeded without contact
    """
    ee_state = p.getLinkState(robot, ee_link)
    cur = list(ee_state[0])
    
    # Build list of obstacle bodies
    contact_bodies = []
    if COLLISION_AWARE_ENABLED and parts:
        contact_bodies = get_obstacle_ids_from_parts(parts)
        if excluded_part and excluded_part in parts:
            excluded_body = parts[excluded_part][0]
            contact_bodies = [b for b in contact_bodies if b != excluded_body]
    
    gripper_body = get_visual_gripper_body()
    
    # First go back to grasp_pos (if we drifted)
    ok1 = approach_linear(
        robot, ee_link, cur, grasp_pos,
        steps=max(10, steps // 2),
        stop_on_contact=True,
        contact_bodies=contact_bodies if contact_bodies else None,
        gripper_body=gripper_body,
        use_compliance=True
    )
    if not ok1:
        print("[RETREAT] Contact while returning to grasp_pos; continuing anyway")
    
    # Then retreat to hover_pos (the main retreat)
    ok2 = approach_linear(
        robot, ee_link, grasp_pos, hover_pos,
        steps=steps,
        stop_on_contact=True,
        contact_bodies=contact_bodies if contact_bodies else None,
        gripper_body=gripper_body,
        use_compliance=True
    )
    
    if ok2:
        print("[RETREAT] Successfully retreated to hover position")
    else:
        print("[RETREAT] Contact during retreat; may need manual recovery")
    
    return ok2


def check_grasp_proximity(robot, ee_link, part_body, max_distance=0.10, tcp_offset_z=0.10,
                          use_simple_vertical=False):
    """Check if the gripper TCP is close enough to grasp the part.
    
    This ensures we only create grasp constraints when the gripper
    is actually near the part, not reaching through geometry.
    
    The TCP (tool center point) is offset from the EE flange - this is
    where the gripper fingers actually meet the object.
    
    Two modes:
    - Orientation-aware (default): Uses proper transform math. Correct for
      angled grasps but can give large distances when wrist is tilted.
    - Simple vertical (use_simple_vertical=True): Assumes overhead grasp,
      just subtracts TCP offset from EE Z. Better for overhead grasps where
      the wrist tilts but we still want to measure vertical proximity.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        part_body: Body ID of the part to grasp
        max_distance: Maximum allowed distance (default 10cm)
        tcp_offset_z: Offset from EE to TCP along tool axis (default 10cm)
        use_simple_vertical: If True, use simple Z offset (for overhead grasps)
        
    Returns:
        Tuple of (is_close_enough, distance)
    """
    import math
    
    ee_state = p.getLinkState(robot, ee_link)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    obj_pos, _ = p.getBasePositionAndOrientation(part_body)
    
    if use_simple_vertical:
        # Simple vertical offset - good for overhead grasps where wrist tilts
        # but we still want to measure how close we are vertically
        tcp_world = [ee_pos[0], ee_pos[1], ee_pos[2] - tcp_offset_z]
    else:
        # Orientation-aware - correct for angled grasps
        tcp_local = [0, 0, -tcp_offset_z]  # Pinch point in EE frame
        tcp_world, _ = p.multiplyTransforms(ee_pos, ee_orn, tcp_local, [0, 0, 0, 1])
    
    # Calculate distance from TCP to object center
    distance = math.sqrt(
        (tcp_world[0] - obj_pos[0])**2 +
        (tcp_world[1] - obj_pos[1])**2 +
        (tcp_world[2] - obj_pos[2])**2
    )
    
    return distance <= max_distance, distance


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
        
        # Use collision-safe motion for approach
        move_ee_safe(robot, ee_link, hover_pos, parts=parts,
                     excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                     steps=80)
    except Exception as e:
        print(f"Warning: Could not move to hover position: {e}")
        step_sim(0.1)
        return
    
    try:
        # Move down to part - with collision checking ON
        work_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
        print(f"    Moving to work position: {work_pos}")
        move_ee_safe(robot, ee_link, work_pos, parts=parts,
                     excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                     steps=60)
    except Exception as e:
        print(f"Warning: Could not move to work position: {e}")
        step_sim(0.1)
        return
    
    # Wiggle action (simulate screwing/unscrewing) - MORE WIGGLES
    print(f"    Working on {part_name}...")
    for i in range(5):  # Increased from 3
        # Just step simulation for visual wiggle effect
        step_sim(0.1)  # Increased from 0.05
    
    try:
        # Return to hover - INCREASED STEPS (collision-safe retreat)
        print(f"    Returning to hover position")
        move_ee_safe(robot, ee_link, hover_pos, parts=parts,
                     excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                     steps=60)
    except Exception as e:
        print(f"Warning: Could not return to hover: {e}")
        step_sim(0.1)


def pick_up_part(robot, ee_link, gripper, open_val, close_val, parts, part_name, original_positions=None):
    """Pick up a part using constraint-based grasping.
    
    This function implements reliable grasping using PyBullet constraints:
    1. Makes the part dynamic (if static) so it can be moved
    2. Opens the gripper
    3. Moves the gripper above the part (hover position - safe approach)
    4. Moves down to the part (grasp position)
    5. Creates a fixed constraint to attach object to gripper
    6. Closes the gripper (for visual feedback)
    7. Lifts the part up (object moves with gripper due to constraint)
    
    The constraint approach ensures the object stays firmly attached during
    motion, unlike friction-based grasping which can slip.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        gripper: List of gripper joint indices (can be empty for KUKA)
        open_val: Gripper open position value
        close_val: Gripper close position value
        parts: Dictionary of part names to (body_id, link_index)
        part_name: Name of the part to pick up
        original_positions: Dict of original part positions for reference
        
    Returns:
        bool: True if pickup was successful, False otherwise
    """
    if part_name not in parts:
        print(f"[PICKUP] Part '{part_name}' not in scene")
        return False
    
    try:
        target_pos, _ = get_pos(parts[part_name])
        
        # Validate position - check if part has been teleported away
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
    
    # Get the part body ID for constraint creation
    part_body, part_link = parts[part_name]
    
    # =========================================================================
    # Build list of contact bodies for collision-aware approach
    # =========================================================================
    # Instead of disabling collisions (which breaks physics), we keep collisions
    # ON and use linear approach with contact detection
    contact_bodies = [pbody for pname, (pbody, _) in parts.items() if pname != part_name]
    
    # =========================================================================
    # Step 0b: Get semantic grasp strategy based on part type and geometry
    # =========================================================================
    grasp_strategy = select_grasp_strategy(part_name, part_body)
    print(f"[PICKUP] Grasp strategy for '{part_name}': {grasp_strategy['approach_dir']} approach")
    
    # =========================================================================
    # Step 0c: Make the part dynamic (if it's static) so it can be moved
    # =========================================================================
    # Check if the part has mass 0 (static) - if so, make it dynamic
    try:
        dynamics_info = p.getDynamicsInfo(part_body, -1)
        current_mass = dynamics_info[0]
        
        if current_mass == 0:
            print(f"[PICKUP] Part '{part_name}' is static (mass=0), making it dynamic...")
            from sim_scene import make_part_dynamic
            
            # IMPORTANT: Save old body ID BEFORE updating part_body
            old_body_id = part_body
            new_body_id = make_part_dynamic(old_body_id, mass=0.5)
            
            # Update the parts dictionary with new body ID
            parts[part_name] = (new_body_id, -1)
            part_body = new_body_id
            
            # Remove OLD body from contact_bodies (it no longer exists)
            # Bug fix: was incorrectly removing new_body_id instead of old_body_id
            contact_bodies = [b for b in contact_bodies if b != old_body_id]
            
            # Need to get new position after recreation
            target_pos, _ = get_pos(parts[part_name])
            
            # Let physics settle briefly
            step_sim(0.1)
            
            # Re-read target_pos after physics settling
            target_pos, _ = get_pos(parts[part_name])
    except Exception as e:
        print(f"[PICKUP] Warning: Could not check/change part mass: {e}")
    
    # =========================================================================
    # Store original part position for reference during approach
    # =========================================================================
    original_part_pos = list(target_pos)
    
    # =========================================================================
    # TCP (Tool Center Point) offset - aligns gripper fingers with object
    # =========================================================================
    # The pinch point of the gripper is offset from the EE flange.
    # This offset positions the fingers around the object, not the flange.
    TCP_OFFSET_Z = 0.10  # 10cm below EE flange is where fingers meet
    
    # =========================================================================
    # Get visual gripper body for collision detection (KUKA only)
    # =========================================================================
    gripper_body = get_visual_gripper_body()
    
    # =========================================================================
    # Step 1: Open gripper and release any previous grasp
    # =========================================================================
    print(f"[PICKUP] Opening gripper...")
    open_gripper(robot, gripper, open_val)
    step_sim(0.2)
    
    # NOTE: Do NOT reset part position here - let physics handle it
    # Resetting fights physics and causes jitter
    
    # =========================================================================
    # Step 2: Compute approach positions using semantic grasp strategy
    # =========================================================================
    # Use grasp strategy to determine approach direction and positions
    hover_height = 0.25  # 25cm clearance
    
    # -------------------------------------------------------------------------
    # 2a: Determine if angled grasp is allowed for this part category
    # -------------------------------------------------------------------------
    cat = part_category(part_name)
    approach_dir = grasp_strategy['approach_dir']
    use_angled = (
        EXECUTE_ANGLED_GRASPS
        and approach_dir != 'above'
        and ANGLED_GRASP_ALLOWLIST.get(cat, False)
    )
    
    if approach_dir != 'above' and not use_angled:
        reason = "globally disabled" if not EXECUTE_ANGLED_GRASPS else f"category '{cat}' not in allowlist"
        print(f"[PICKUP] Angled grasp ({approach_dir}) blocked: {reason} → using overhead")
        # Force overhead grasp for safety - full valid strategy dict
        grasp_strategy = {
            'approach_dir': 'above',
            'approach_offset': [0, 0, 0.15],
            'grasp_offset': [0, 0, 0.10],
            'ee_orientation': None,
            'gripper_open': 0.0
        }
    
    # -------------------------------------------------------------------------
    # 2b: For side grasps, pick best side (+Y or -Y) based on clearance
    # -------------------------------------------------------------------------
    side_sign = 1  # default: +Y
    if grasp_strategy['approach_dir'] == 'side' and COLLISION_AWARE_ENABLED:
        # Score both sides - pass body ID, not part name
        excluded_body = parts[part_name][0]
        score_pos = score_side_clearance(robot, ee_link, target_pos, parts, excluded_body=excluded_body, sign=+1)
        score_neg = score_side_clearance(robot, ee_link, target_pos, parts, excluded_body=excluded_body, sign=-1)
        side_sign = +1 if score_pos <= score_neg else -1
        print(f"[PICKUP] Side clearance scores: +Y={score_pos:.3f}, -Y={score_neg:.3f} → using {'+Y' if side_sign == 1 else '-Y'}")
    
    # -------------------------------------------------------------------------
    # 2c: Compute hover / grasp / orientation
    # -------------------------------------------------------------------------
    hover_pos, grasp_pos, ee_orn = compute_approach_positions(
        target_pos, grasp_strategy, 
        hover_height=hover_height, 
        tcp_offset=TCP_OFFSET_Z,
        side_sign=side_sign
    )
    
    try:
        print(f"[PICKUP] Moving to hover position: {hover_pos}")
        
        # Use collision-safe motion to hover (checking against all parts except target)
        # For angled grasps, if collision detected, fall back to overhead
        if COLLISION_AWARE_ENABLED:
            would_collide, reason = check_motion_collision(
                robot, ee_link, hover_pos, parts, excluded_part=part_name
            )
            if would_collide:
                if ee_orn is not None:
                    print(f"[PICKUP] Angled hover blocked: {reason} → falling back to overhead")
                    ee_orn = None  # Reset to overhead grasp
                    hover_pos = [target_pos[0], target_pos[1], target_pos[2] + hover_height]
                    grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + TCP_OFFSET_Z]
                else:
                    print(f"[PICKUP] Warning: Hover motion may collide: {reason}")
        
        # Use orientation-aware motion if we have a target orientation
        if ee_orn is not None:
            print(f"[PICKUP] Using orientation-constrained hover")
            move_ee_pose(robot, ee_link, hover_pos, ee_orn, steps=100)
        else:
            move_ee_safe(robot, ee_link, hover_pos, parts=parts, 
                         excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                         steps=100)
    except Exception as e:
        print(f"[PICKUP] Could not move to hover: {e}")
        step_sim(0.2)
        return False
    
    step_sim(0.2)
    
    # NOTE: Do NOT reset part position here - let physics handle it
    # Only reset once, right before creating the constraint
    
    # =========================================================================
    # Step 3: Linear approach to grasp position (with compliance control)
    # =========================================================================
    # Use linear interpolation with contact checking and compliance for
    # realistic behavior. Compliance reduces forces near contact.
    
    # Get current EE position for linear approach
    ee_state = p.getLinkState(robot, ee_link)
    current_ee_pos = list(ee_state[0])
    
    print(f"[PICKUP] Linear approach to grasp position: {grasp_pos}")
    approach_success = approach_linear(
        robot, ee_link, 
        current_ee_pos, grasp_pos,
        steps=25,
        stop_on_contact=True,
        contact_bodies=contact_bodies,
        gripper_body=gripper_body,
        use_compliance=True  # Enable soft contact behavior
    )
    
    if not approach_success:
        print(f"[PICKUP] Contact during approach - trying from different angle")
        # Try approaching from a lateral offset (use strategy hint)
        if grasp_strategy['approach_dir'] == 'front':
            offset_pos = [grasp_pos[0] + 0.08, grasp_pos[1], grasp_pos[2]]
        elif grasp_strategy['approach_dir'] == 'side':
            offset_pos = [grasp_pos[0], grasp_pos[1] + 0.08, grasp_pos[2]]
        else:
            offset_pos = [grasp_pos[0] + 0.05, grasp_pos[1], grasp_pos[2] + 0.05]
        
        # Use collision-safe motion for retry
        move_ee_safe(robot, ee_link, offset_pos, parts=parts,
                     excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                     steps=50)
        ee_state = p.getLinkState(robot, ee_link)
        current_ee_pos = list(ee_state[0])
        approach_success = approach_linear(
            robot, ee_link,
            current_ee_pos, grasp_pos,
            steps=20,
            stop_on_contact=True,
            contact_bodies=contact_bodies,
            gripper_body=gripper_body,
            use_compliance=True
        )
    
    # Abort if both approach attempts failed - prevents ghost grasping
    if not approach_success:
        print(f"[PICKUP] Approach failed after retries - aborting grasp")
        return False
    
    # =========================================================================
    # Step 4: Proximity check before creating constraint
    # =========================================================================
    # Only attach if gripper TCP is actually close to the part.
    # 10cm threshold prevents ghost attachment while allowing for IK tolerances.
    # 
    # For overhead grasps (when ee_orn is None), use simple
    # vertical TCP calculation since the wrist tilts but we're still grasping
    # from above. For angled grasps, use orientation-aware calculation.
    GRASP_PROXIMITY_THRESHOLD = 0.10  # 10cm - tight threshold
    use_simple = (ee_orn is None)  # Simple vertical for overhead grasps
    is_close, distance = check_grasp_proximity(
        robot, ee_link, part_body, 
        max_distance=GRASP_PROXIMITY_THRESHOLD,
        use_simple_vertical=use_simple
    )
    
    if not is_close:
        print(f"[PICKUP] Too far to grasp ({distance:.3f}m > {GRASP_PROXIMITY_THRESHOLD}m) - aborting")
        return False
    
    print(f"[PICKUP] Proximity check passed: {distance:.3f}m from part")
    
    # =========================================================================
    # Step 5: Create constraint to attach object to end-effector
    # =========================================================================
    # Use current part position (not original) to avoid snap if physics drifted
    current_part_pos, current_part_orn = p.getBasePositionAndOrientation(part_body)
    p.resetBasePositionAndOrientation(part_body, current_part_pos, current_part_orn)
    
    # This is the key step - the constraint "welds" the object to the gripper
    print(f"[PICKUP] Creating grasp constraint for {part_name}...")
    constraint_id = create_grasp_constraint(robot, ee_link, part_body, part_link)
    
    if constraint_id is None:
        print(f"[PICKUP] Warning: Could not create grasp constraint, continuing anyway")
    
    step_sim(0.1)
    
    # =========================================================================
    # Step 6: Close gripper (visual feedback)
    # =========================================================================
    if gripper and len(gripper) > 0:
        print(f"[PICKUP] Closing gripper around {part_name}...")
        close_gripper(robot, gripper, close_val)
    else:
        print(f"[PICKUP] No physical gripper - object attached via constraint")
        step_sim(0.2)
    
    step_sim(0.3)
    
    # =========================================================================
    # Step 7: Lift the part
    # =========================================================================
    # Lift to a safe height above the chair
    lift_height = 0.45  # 45cm above original position
    try:
        lift_pos = [target_pos[0], target_pos[1], target_pos[2] + lift_height]
        print(f"[PICKUP] Lifting {part_name} to {lift_pos}...")
        
        # Use collision-safe motion during lift - exclude the grasped part
        move_ee_safe(robot, ee_link, lift_pos, parts=parts,
                     excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                     steps=100)
    except Exception as e:
        print(f"[PICKUP] Could not lift part: {e}")
        step_sim(0.2)
        return False
    
    step_sim(0.3)
    print(f"[PICKUP] Successfully picked up {part_name}")
    return True


def place_part(robot, ee_link, gripper, open_val, close_val, parts, part_name, drop_zone=None):
    """Place a part at a drop zone location using constraint-based release.
    
    This function:
    1. Moves the grasped object to the drop zone position
    2. Releases the grasp constraint (object becomes free-floating)
    3. Opens the gripper (visual feedback)
    4. Retracts the robot upward to clear the placed object
    
    The constraint is released first, then the gripper opens. This ensures
    the object stays at the drop location and doesn't get dragged by the gripper.
    
    Args:
        robot: Robot body ID
        ee_link: End-effector link index
        gripper: List of gripper joint indices (can be empty for KUKA)
        open_val: Gripper open position value
        close_val: Gripper close position value  
        parts: Dictionary of parts (not used but kept for consistency)
        part_name: Name of part being placed (for logging)
        drop_zone: Target position [x, y, z]. If None, uses default.
        
    Returns:
        bool: True if placement was successful
    """
    if drop_zone is None:
        drop_zone = [1.0, 0.0, 0.4]  # Default drop zone (to the side of chair)
    
    # =========================================================================
    # Step 1: Move to drop zone position (collision-aware)
    # =========================================================================
    try:
        print(f"[PLACE] Moving {part_name} to drop zone: {drop_zone}")
        
        # Use collision-safe motion to drop zone, excluding the held part
        move_ee_safe(robot, ee_link, drop_zone, parts=parts,
                     excluded_part=part_name, use_collision_check=COLLISION_AWARE_ENABLED,
                     steps=100)
    except Exception as e:
        print(f"[PLACE] Could not move to drop zone: {e}")
        step_sim(0.2)
        return False
    
    step_sim(0.3)
    
    # =========================================================================
    # Step 2: Release the grasp constraint FIRST
    # =========================================================================
    # Important: Release constraint before opening gripper so object stays in place
    print(f"[PLACE] Releasing grasp constraint...")
    release_grasp()
    step_sim(0.2)
    
    # =========================================================================
    # Step 3: Open gripper (visual feedback for physical grippers)
    # =========================================================================
    if gripper and len(gripper) > 0:
        print(f"[PLACE] Opening gripper to release {part_name}...")
        open_gripper(robot, gripper, open_val)
    else:
        print(f"[PLACE] Released {part_name} (no physical gripper)")
        step_sim(0.1)
    
    step_sim(0.2)
    
    # =========================================================================
    # Step 4: Retract using mirrored retreat (reverses approach path)
    # =========================================================================
    try:
        # Build contact bodies for retreat (all parts except the released one)
        # Use mirrored retreat: from drop_zone, ascend vertically by 0.2m
        hover_pos = [drop_zone[0], drop_zone[1], drop_zone[2] + 0.2]
        print(f"[PLACE] Mirrored retreat to {hover_pos}...")
        
        retreat_mirrored(
            robot, ee_link,
            grasp_pos=drop_zone,
            hover_pos=hover_pos,
            parts=parts,
            excluded_part=None,  # Part is released, don't exclude
            steps=40
        )
    except Exception as e:
        print(f"[PLACE] Could not retract: {e}")
    
    step_sim(0.2)
    print(f"[PLACE] Successfully placed {part_name}")
    return True


def spawn_replacement_part(parts, original_part_name, original_positions, spawn_offset=[0.2, 0.2, 0.3]):
    """Spawn a new replacement part near the original position.
    
    This creates a new DYNAMIC (movable) part that can be picked up by the gripper.
    The part has mass so it can be physically manipulated.
    
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
            color = [0.2, 0.8, 0.2, 1]  # Green (new part)
            mass = 0.5  # 500g
        elif original_part_name == "seat":
            size = [0.45, 0.45, 0.05]
            color = [0.2, 0.8, 0.2, 1]
            mass = 2.0  # 2kg
        elif original_part_name == "back":
            size = [0.05, 0.45, 0.5]
            color = [0.2, 0.8, 0.2, 1]
            mass = 1.5  # 1.5kg
        elif "armrest" in original_part_name:
            size = [0.6, 0.05, 0.05]
            color = [0.2, 0.8, 0.2, 1]
            mass = 0.3  # 300g
        else:
            print(f"[SPAWN] Unknown part type: {original_part_name}")
            return False
        
        # Calculate spawn position (offset from original, visible to user)
        spawn_pos = [
            original_pos[0] + spawn_offset[0],
            original_pos[1] + spawn_offset[1],
            original_pos[2] + spawn_offset[2]
        ]
        
        # Create a DYNAMIC replacement part that can be picked up
        from sim_scene import create_dynamic_block
        replacement_body = create_dynamic_block(size, spawn_pos, color, mass=mass)
        
        # Add to parts dict with a special name
        replacement_name = f"{original_part_name}_replacement"
        parts[replacement_name] = (replacement_body, -1)
        
        # Store the replacement's original position for later
        original_positions[replacement_name] = spawn_pos
        
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
                        move_ee_safe(robot, ee_link, install_pos, parts=parts,
                                     excluded_part=replacement_name, use_collision_check=COLLISION_AWARE_ENABLED,
                                     steps=80)
                    except Exception as e:
                        print(f"  -> Could not move to installation position: {e}")
                    
                    step_sim(0.2)
                    
                    # Release grasp constraint first, then open gripper
                    print(f"  -> Releasing replacement part at installation position")
                    release_grasp()  # Remove constraint so object stays in place
                    
                    if gripper and len(gripper) > 0:
                        open_gripper(robot, gripper, open_val)
                    else:
                        step_sim(0.1)
                    
                    step_sim(0.2)
                    
                    # =========================================================
                    # PROPER REPLACEMENT LOGIC:
                    # 1. Place the replacement body at the original position
                    # 2. Update parts dict so the replacement IS the new part
                    # 3. Remove the replacement_name entry (now it's "part")
                    # =========================================================
                    try:
                        # Get the old damaged body and replacement body
                        old_body, old_link = parts[part]
                        new_body, new_link = parts[replacement_name]
                        
                        # Move replacement to original position (it becomes the new part)
                        p.resetBasePositionAndOrientation(new_body, original_pos, [0, 0, 0, 1])
                        
                        # Change color to normal (undamaged)
                        p.changeVisualShape(new_body, -1, rgbaColor=(0.6, 0.4, 0.2, 1))
                        
                        # Update the parts dictionary: the chair now has the replacement
                        parts[part] = (new_body, -1)
                        
                        # Remove the old damaged body from scene (teleport away)
                        p.resetBasePositionAndOrientation(old_body, [10, 10, -10], [0, 0, 0, 1])
                        
                        # Remove the replacement_name key (it's now just "part")
                        del parts[replacement_name]
                        
                        # Clean up original_positions to prevent stale position snapping
                        if original_positions and replacement_name in original_positions:
                            del original_positions[replacement_name]
                        
                        print(f"  -> {part} successfully replaced! (new body {new_body})")
                    except Exception as e:
                        print(f"  -> Error during replacement install: {e}")
                    
                    step_sim(0.2)
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
                        move_ee_safe(robot, ee_link, install_pos, parts=parts,
                                     excluded_part=replacement_name, use_collision_check=COLLISION_AWARE_ENABLED,
                                     steps=80)
                    except Exception as e:
                        print(f"  -> Could not move to installation position: {e}")
                    
                    step_sim(0.2)
                    
                    # Release grasp constraint first, then open gripper
                    print(f"  -> Releasing replacement part at installation position")
                    release_grasp()  # Remove constraint so object stays in place
                    
                    if gripper and len(gripper) > 0:
                        open_gripper(robot, gripper, open_val)
                    else:
                        step_sim(0.1)
                    
                    step_sim(0.2)
                    
                    # =========================================================
                    # PROPER INSTALL LOGIC:
                    # 1. Place the replacement body at the original position
                    # 2. Update parts dict so the replacement IS the new part
                    # 3. Remove the replacement_name entry
                    # =========================================================
                    try:
                        # Get the old damaged body and replacement body
                        old_body, old_link = parts[part]
                        new_body, new_link = parts[replacement_name]
                        
                        # Move replacement to original position (it becomes the new part)
                        p.resetBasePositionAndOrientation(new_body, original_pos, [0, 0, 0, 1])
                        
                        # Change color to normal (undamaged)
                        p.changeVisualShape(new_body, -1, rgbaColor=(0.6, 0.4, 0.2, 1))
                        
                        # Update the parts dictionary: the chair now has the replacement
                        parts[part] = (new_body, -1)
                        
                        # Remove the old damaged body from scene (teleport away)
                        p.resetBasePositionAndOrientation(old_body, [10, 10, -10], [0, 0, 0, 1])
                        
                        # Remove the replacement_name key (it's now just "part")
                        del parts[replacement_name]
                        
                        # Clean up original_positions to prevent stale position snapping
                        if original_positions and replacement_name in original_positions:
                            del original_positions[replacement_name]
                        
                        print(f"  -> {part} successfully installed! (new body {new_body})")
                    except Exception as e:
                        print(f"  -> Error during install: {e}")
                    
                    step_sim(0.2)
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
    
    elif action == "clean":
        print(f"  -> Cleaning {part}...")
        recolor(parts[part], (0.5, 0.8, 1, 1))  # Light blue (cleaning)
        show_working_animation(robot, ee_link, parts, part, original_positions=original_positions)
        recolor(parts[part], (0.6, 0.4, 0.2, 1))  # Restore
        step_sim(0.5)

    else:
        # Generic action
        print(f"  -> Processing {part} ({action})...")
        if part in parts:
            safe_call(show_working_animation, robot, ee_link, parts, part, original_positions=original_positions, timeout=12)
        step_sim(0.5)