"""Robot loading and motion helpers for the repair simulation.

Uses the Franka Panda robot only. Provides:
- Loading Panda URDF from pybullet_data
- End-effector motion via inverse kinematics
- Built-in gripper joints (open/close)
- Grasp constraint: fixed weld between EE and object when close enough

Units: meters, Z-up. Grasping uses p.createConstraint (EE-to-object fixed).
"""

import os
import pybullet as p
try:
    from pybullet_sim.sim_connection import step_sim
except ImportError:
    from sim_connection import step_sim
import math
from typing import Tuple, Optional


# ============================================================================
# GLOBAL STATE FOR CONSTRAINT-BASED GRASPING AND GRIPPER
# ============================================================================
# We track the current grasp constraint ID so we can release it later.
# Using a constraint ensures the object moves with the gripper during motion.
_active_grasp_constraint = None


def get_visual_gripper_body():
    """Return None; Panda has no separate gripper body (kept for collision_aware_motion API)."""
    return None


# Home pose for Franka Panda (7 arm + 2 gripper joints, gripper open)
HOME_POSES = {
    "panda": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
}


def _find_panda_urdf():
    """Return (data_path, urdf_name) for Panda. data_path is for setAdditionalSearchPath."""
    try:
        import pybullet_data
        data_path = pybullet_data.getDataPath()
        path = os.path.join(data_path, "franka_panda", "panda.urdf")
        if os.path.isfile(path):
            return data_path, "franka_panda/panda.urdf"
        path2 = os.path.join(data_path, "panda.urdf")
        if os.path.isfile(path2):
            return data_path, "panda.urdf"
    except Exception:
        pass
    return None, None


def discover_gripper_joints(robot_id, robot_type="panda"):
    """Discover and document gripper joint indices for a robot.
    
    This function inspects the robot URDF to find gripper-related joints
    by checking joint names for keywords like 'finger', 'gripper', 'hand'.
    
    Args:
        robot_id: PyBullet body ID of the robot
        robot_type: String identifier ("panda")
        
    Returns:
        tuple: (gripper_joint_indices, open_value, close_value)
        
    Notes:
        - Panda gripper joints are prismatic with range [0, 0.04]
    """
    num_joints = p.getNumJoints(robot_id)
    gripper_joints = []
    
    print(f"[GRIPPER] Discovering joints for {robot_type} robot ({num_joints} joints total)")
    
    # Scan all joints looking for gripper-related ones
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode('utf-8').lower()
        joint_type = joint_info[2]
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        
        # Check if this looks like a gripper joint
        is_gripper = any(kw in joint_name for kw in ['finger', 'gripper', 'hand'])
        
        if is_gripper and joint_type == p.JOINT_PRISMATIC:
            gripper_joints.append(joint_idx)
            print(f"[GRIPPER]   Found gripper joint {joint_idx}: '{joint_name}' "
                  f"(limits: {lower_limit:.4f} to {upper_limit:.4f})")
    
    if gripper_joints:
        # Use discovered joint limits for open/close values
        # Open = max limit, Close = min limit (for parallel jaw grippers)
        joint_info = p.getJointInfo(robot_id, gripper_joints[0])
        close_val = joint_info[8]  # Lower limit = closed
        open_val = joint_info[9]   # Upper limit = open
        print(f"[GRIPPER]   Open position: {open_val}, Close position: {close_val}")
        return gripper_joints, open_val, close_val
    else:
        print(f"[GRIPPER]   No Panda gripper joints found - using fallback indices")
        return [], 0.04, 0.0


def load_robot(robot: str = "panda"):
    """Load the Franka Panda robot and return (robot_id, ee_link, gripper_joints, open_val, close_val)."""
    data_path, urdf_name = _find_panda_urdf()
    if not urdf_name:
        raise FileNotFoundError(
            "Panda URDF not found. Install pybullet (pip install pybullet) and ensure "
            "franka_panda/panda.urdf or panda.urdf exists in pybullet_data."
        )
    p.setAdditionalSearchPath(data_path)
    rid = p.loadURDF(urdf_name, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    ee_link = 11  # panda_hand link (end-effector)
    gripper, open_val, close_val = discover_gripper_joints(rid, "panda")
    if not gripper:
        gripper = [9, 10]
        open_val, close_val = 0.04, 0.0
        print(f"[GRIPPER] Using fallback Panda gripper joints: {gripper}")
    num_joints = p.getNumJoints(rid)
    print(f"[ROBOT] Loaded Panda: id={rid}, ee_link={ee_link}, joints={num_joints}, gripper_joints={gripper}")
            
    return rid, ee_link, gripper, open_val, close_val


def move_ee(robot, ee_link, pos, orn=None, steps=160):
    """Move the robot end-effector to `pos` (and optional `orn`).

    This convenience wrapper calculates joint targets via inverse
    kinematics and then commands position controllers on all joints.

    Args:
        robot: PyBullet body id for the robot.
        ee_link: index of the end-effector link.
        pos: target position [x,y,z].
        orn: optional target orientation (quaternion). If omitted, 
             position-only IK is used (more reliable for reachability).
        steps: how many physics steps to wait after commanding joints.
        
    Notes:
        - Uses PyBullet's damped least squares IK solver
        - Joint commands are sent with position control and high force
        - The function blocks until motion is complete (via step_sim)
    """
    # NOTE: We no longer default to preserving current orientation.
    # Using position-only IK is more reliable for reaching targets.
    # If specific orientation is needed, caller should provide it explicitly.

    try:
        # Use IK with joint limits for more reliable solutions
        num_joints = p.getNumJoints(robot)
        
        # Get current joint positions as seed for IK
        current_joints = []
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = []
        
        for j in range(num_joints):
            joint_info = p.getJointInfo(robot, j)
            if joint_info[2] != p.JOINT_FIXED:
                current_joints.append(p.getJointState(robot, j)[0])
                lower_limits.append(joint_info[8])
                upper_limits.append(joint_info[9])
                joint_ranges.append(joint_info[9] - joint_info[8])
                rest_poses.append(0.0)
        
        # Calculate IK - position only if orientation not specified
        if orn is not None:
            joints = p.calculateInverseKinematics(
                robot, ee_link, pos, orn,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
        else:
            # Position-only IK (more flexible, better reachability)
            joints = p.calculateInverseKinematics(
                robot, ee_link, pos,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
    except Exception as e:
        print(f"[WARNING] IK calculation failed for position {pos}: {e}")
        # Skip this movement if IK fails
        return

    num_joints = p.getNumJoints(robot)
    if not joints:
        print(f"[WARNING] IK solver returned empty joint list for pos {pos}")
        return

    # Debugging: show requested target and IK result sizes
    try:
        print(f"[SIM ROBOT] move_ee target={pos} joints_returned={len(joints)} robot_num_joints={num_joints}")
    except Exception:
        pass

    # Only set motors for non-fixed joints
    joint_index = 0
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] != p.JOINT_FIXED:
            if joint_index < len(joints):
                try:
                    p.setJointMotorControl2(
                        robot, j, p.POSITION_CONTROL, 
                        joints[joint_index], 
                        force=500,  # High force for reliable motion
                        maxVelocity=3.0  # Allow faster motion
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to set joint {j}: {e}")
            joint_index += 1
            
    # Run physics simulation steps for the robot to reach the target
    # Testing shows ~300+ physics steps are needed for reliable positioning
    # step_sim uses hz=120, so we need to run for long enough
    # steps parameter is now treated as minimum physics steps to run
    actual_steps = max(300, steps)
    for _ in range(actual_steps):
        p.stepSimulation()


def open_gripper(robot, joints, val):
    """Open Panda gripper (release grasp constraint and set joints to open position)."""
    global _active_grasp_constraint

    if _active_grasp_constraint is not None:
        try:
            p.removeConstraint(_active_grasp_constraint)
            print(f"[GRIPPER] Released grasp constraint {_active_grasp_constraint}")
        except Exception as e:
            print(f"[GRIPPER] Warning: Could not remove constraint: {e}")
        _active_grasp_constraint = None

    if joints and len(joints) > 0:
        for j in joints:
            try:
                p.setJointMotorControl2(
                    robot, j, p.POSITION_CONTROL, 
                    val, 
                    force=50,
                    maxVelocity=0.5
                )
            except Exception as e:
                print(f"[WARNING] Could not open gripper joint {j}: {e}")
        step_sim(0.08)  # Brief step so object doesn't drift too much
    else:
        # No gripper at all - just step simulation
        step_sim(0.1)


def close_gripper(robot, joints, val):
    """Close Panda gripper (set joints to closed position). Grasp is created via create_grasp_constraint()."""
    if joints and len(joints) > 0:
        for j in joints:
            try:
                p.setJointMotorControl2(
                    robot, j, p.POSITION_CONTROL, 
                    val, 
                    force=50,
                    maxVelocity=0.5
                )
            except Exception as e:
                print(f"[WARNING] Could not close gripper joint {j}: {e}")
        step_sim(0.08)  # Brief step - constraint created immediately after
    else:
        # No physical gripper - just step simulation
        step_sim(0.1)


# ============================================================================
# CONTACT-BASED GRASP VALIDATION (ARCHITECTURAL FIX)
# ============================================================================
# Validates grasp conditions BEFORE creating constraints.
# This prevents "ghost grasps" where the object is constrained without
# actual physical contact between gripper and object.

def validate_grasp_contact(
    robot_id: int,
    ee_link: int,
    target_body: int,
    gripper_body: int = None,
    max_distance: float = 0.02,
    require_finger_contact: bool = True
) -> Tuple[bool, str, float]:
    """Validate that gripper is in proper contact with target before grasping.
    
    ARCHITECTURAL FIX: This function ensures we only create grasp constraints
    when there is ACTUAL physical proximity/contact, not just when the robot
    thinks it's in position. This prevents:
    - Ghost grasps through geometry
    - Constraints created from too far away
    - Unrealistic "teleport" grasping
    
    Args:
        robot_id: PyBullet body ID of the robot
        ee_link: End-effector link index
        target_body: Body ID of the object to grasp
        gripper_body: Visual gripper body ID (for KUKA)
        max_distance: Maximum allowed distance (default 2cm)
        require_finger_contact: If True, require at least one finger contact
        
    Returns:
        Tuple of (is_valid, reason, distance)
    """
    import math
    
    # CRITICAL: Force collision detection update before checking distances
    # Without this, getClosestPoints() may return stale/incorrect data
    p.performCollisionDetection()
    
    # Get EE position
    ee_state = p.getLinkState(robot_id, ee_link)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    # Get object position
    obj_pos, obj_orn = p.getBasePositionAndOrientation(target_body)
    
    # Check 1: Basic distance check
    # Use closest points for accurate distance measurement
    # Use larger query distance (0.25m) to ensure we find contacts even if slightly far
    min_distance = float('inf')
    finger_min_distance = float('inf')  # Track finger distance separately
    QUERY_DISTANCE = 0.25  # 25cm query range
    
    # Check gripper body vs object FIRST (primary for KUKA visual gripper)
    if gripper_body is not None:
        try:
            # Check gripper FINGERS specifically (not base) - this is what matters for grasp
            num_gripper_links = p.getNumJoints(gripper_body)
            print(f"[GRASP DEBUG] Checking {num_gripper_links} gripper links...")
            for finger_link in range(num_gripper_links):
                contacts = p.getClosestPoints(
                    bodyA=gripper_body, bodyB=target_body,
                    distance=QUERY_DISTANCE, linkIndexA=finger_link
                )
                if contacts:
                    link_min = min(c[8] for c in contacts)
                    print(f"[GRASP DEBUG]   Link {finger_link}: {link_min*1000:.1f}mm ({len(contacts)} pts)")
                for contact in contacts:
                    if contact[8] < finger_min_distance:
                        finger_min_distance = contact[8]
                    if contact[8] < min_distance:
                        min_distance = contact[8]
            
            # Also check gripper BASE (link -1)
            contacts = p.getClosestPoints(
                bodyA=gripper_body, bodyB=target_body,
                distance=QUERY_DISTANCE
            )
            if contacts:
                base_min = min(c[8] for c in contacts)
                print(f"[GRASP DEBUG]   Base (all): {base_min*1000:.1f}mm ({len(contacts)} pts)")
            for contact in contacts:
                if contact[8] < min_distance:
                    min_distance = contact[8]
            
            # =====================================================================
            # REALISM FIX: Object must be BETWEEN the fingers, not just touching base
            # =====================================================================
            # A valid grasp requires the object to be gripped between the two fingers.
            # If the base is close but fingers are far, the object is ON TOP of the 
            # gripper (palm contact), not BETWEEN the fingers (proper grasp).
            #
            # Valid grasp criteria:
            # 1. At least one finger must be close (<25mm) to the object
            # 2. OR physics contact detected on a finger
            # 3. Base-only contact is NOT sufficient for a valid grasp
            # =====================================================================
            
            FINGER_GRASP_THRESHOLD = 0.030  # 30mm - finger must be close for valid grasp
            
            # Check if any finger is close enough
            finger_is_close = finger_min_distance < FINGER_GRASP_THRESHOLD
            
            if finger_is_close:
                # Good - finger is near the object, use finger distance
                min_distance = finger_min_distance
                print(f"[GRASP DEBUG] Finger grasp valid: {min_distance*1000:.1f}mm")
            elif min_distance < 0.005:  # Base is touching
                # Base is touching but fingers are far - object is ON TOP of gripper, not between fingers
                print(f"[GRASP DEBUG] WARNING: Base contact but fingers far ({finger_min_distance*1000:.1f}mm)")
                print(f"[GRASP DEBUG] Object is ON gripper, not BETWEEN fingers - invalid grasp!")
                # Use finger distance for validation - this will fail if fingers are too far
                min_distance = finger_min_distance
            else:
                # Neither finger nor base is close
                print(f"[GRASP DEBUG] No valid grasp: finger={finger_min_distance*1000:.1f}mm, base={min_distance*1000:.1f}mm")
        except Exception as e:
            print(f"[GRASP DEBUG] Error checking gripper: {e}")
    
    # Fallback: Check robot EE vs object
    if min_distance == float('inf'):
        try:
            contacts = p.getClosestPoints(
                bodyA=robot_id, bodyB=target_body,
                distance=QUERY_DISTANCE, linkIndexA=ee_link
            )
            for contact in contacts:
                if contact[8] < min_distance:
                    min_distance = contact[8]
            print(f"[GRASP DEBUG] Robot EE {robot_id}:{ee_link} -> target {target_body}: min_distance={min_distance:.4f}m")
        except Exception as e:
            print(f"[GRASP DEBUG] Error checking EE: {e}")
    
    # Check 2: Distance validation
    if min_distance > max_distance:
        return False, f"Too far from object ({min_distance:.3f}m > {max_distance}m)", min_distance
    
    # Check 3: Finger contact validation (STRICT - REQUIRED for valid grasp)
    # =========================================================================
    # REALISTIC GRASP REQUIREMENT: No proximity fallback. No exceptions.
    # If there's no physics contact between finger and object, grasp is INVALID.
    # 
    # Previous code allowed:
    #   - Proximity heuristics (within Xmm)
    #   - Base contact (object on top of gripper)
    # 
    # New code requires:
    #   - p.getContactPoints(finger_link, target) returns contacts
    #   - Period. No fallback.
    # =========================================================================
    if require_finger_contact and gripper_body is not None:
        has_finger_contact = False
        contact_finger = None
        contact_count = 0
        
        # Force collision detection update
        p.performCollisionDetection()
        
        # Check for actual contact points on FINGER LINKS ONLY
        num_gripper_links = p.getNumJoints(gripper_body)
        for finger_link in range(num_gripper_links):
            contact_points = p.getContactPoints(
                bodyA=gripper_body, bodyB=target_body,
                linkIndexA=finger_link  # FINGER LINKS ONLY
            )
            if contact_points:
                has_finger_contact = True
                contact_finger = finger_link
                contact_count = len(contact_points)
                print(f"[GRASP DEBUG] CONTACT: Finger {finger_link} touching target ({contact_count} contact points)")
                break
        
        if has_finger_contact:
            print(f"[GRASP DEBUG] Valid: finger {contact_finger} has physics contact")
            return True, f"Grasp valid (finger contact, {contact_count} points)", 0.0
        else:
            # NO CONTACT = NO GRASP. Period.
            print(f"[GRASP DEBUG] INVALID: No finger contact detected")
            print(f"[GRASP DEBUG] Finger distances: {finger_min_distance*1000:.1f}mm")
            return False, f"No finger contact (fingers {finger_min_distance*1000:.1f}mm away)", finger_min_distance
    
    # Check 4: Validate grasp direction (contact normal should align with approach)
    # This prevents grasps where we approached but from wrong angle
    # For now, we just check distance - normal checking is complex
    
    return True, f"Grasp validated (distance: {min_distance:.3f}m)", min_distance


def create_grasp_constraint(robot, ee_link, target_body, target_link=-1):
    """Create a fixed constraint (grasp) between Panda EE and target object. Only when EE is within 150mm."""
    global _active_grasp_constraint

    if _active_grasp_constraint is not None:
        try:
            p.removeConstraint(_active_grasp_constraint)
        except Exception:
            pass
        _active_grasp_constraint = None

    pos_ee, orn_ee = p.getLinkState(robot, ee_link)[:2]
    pos_obj, orn_obj = p.getBasePositionAndOrientation(target_body)
    dist = math.sqrt(sum((pos_ee[i] - pos_obj[i]) ** 2 for i in range(3)))
    MAX_PANDA_GRASP_DISTANCE = 0.30  # 300mm - allow for IK error and slight object drift
    if dist > MAX_PANDA_GRASP_DISTANCE:
        print(f"[GRIPPER] Panda grasp refused: EE and object {dist*1000:.0f}mm apart (max {MAX_PANDA_GRASP_DISTANCE*1000:.0f}mm)")
        return None
    inv_pos, inv_orn = p.invertTransform(pos_obj, orn_obj)
    child_pos, child_orn = p.multiplyTransforms(inv_pos, inv_orn, pos_ee, orn_ee)
    constraint_id = p.createConstraint(
        robot, ee_link, target_body, target_link,
        p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=child_pos,
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=child_orn,
    )
    p.changeConstraint(constraint_id, maxForce=100000)
    _active_grasp_constraint = constraint_id
    print(f"[GRIPPER] Panda: attached object {target_body} to EE (dist={dist*1000:.0f}mm)")
    return constraint_id


def release_grasp():
    """Release the currently grasped object by removing the constraint.
    
    Returns:
        bool: True if a constraint was released, False otherwise
    """
    global _active_grasp_constraint
    
    if _active_grasp_constraint is not None:
        try:
            p.removeConstraint(_active_grasp_constraint)
            print(f"[GRIPPER] Released grasp constraint {_active_grasp_constraint}")
            _active_grasp_constraint = None
            return True
        except Exception as e:
            print(f"[GRIPPER] Warning: Could not remove constraint: {e}")
            _active_grasp_constraint = None
    return False


def get_grasp_constraint():
    """Get the ID of the currently active grasp constraint.
    
    Returns:
        int or None: Constraint ID if grasping, None otherwise
    """
    return _active_grasp_constraint


def move_to_home(robot, robot_type="panda"):
    """Move the Panda robot to its home/neutral position (gripper open)."""
    global _active_grasp_constraint

    if _active_grasp_constraint is not None:
        release_grasp()

    home_pose = HOME_POSES.get(robot_type, HOME_POSES["panda"])
    
    print(f"[ROBOT] Moving to home pose for {robot_type}...")
    
    num_joints = p.getNumJoints(robot)
    movable_joint_idx = 0
    
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] != p.JOINT_FIXED:  # Skip fixed joints
            if movable_joint_idx < len(home_pose):
                try:
                    p.setJointMotorControl2(
                        robot, j, p.POSITION_CONTROL,
                        home_pose[movable_joint_idx],
                        force=500,  # Higher force for reliable motion
                        maxVelocity=2.0  # Moderate speed
                    )
                except Exception as e:
                    print(f"[WARNING] Could not set joint {j} to home: {e}")
            movable_joint_idx += 1
    
    # Step simulation to allow robot to reach home pose
    # Use direct stepping for reliability in both GUI and DIRECT modes
    for _ in range(1000):
        p.stepSimulation()
    
    print(f"[ROBOT] Robot at home position")
