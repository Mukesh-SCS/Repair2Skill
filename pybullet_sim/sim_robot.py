"""Robot loading and simple motion helpers for the demo.

This module provides a minimal abstraction for loading a robot model,
moving its end-effector using inverse kinematics, and operating a
gripper if available. The functions are intentionally small and
well-documented to make the simulation flow easy to follow.

GRIPPER IMPLEMENTATION NOTES:
-----------------------------
For physical grasping, we use PyBullet constraints (p.createConstraint)
instead of relying on friction-based grasping. This provides reliable
object attachment during robot motion. The constraint acts as a "weld"
between the end-effector and the grasped object.

Supported robots:
- Franka Panda: Uses joints 9 and 10 (panda_finger_joint1, panda_finger_joint2)
  with open position = 0.04m and close position = 0.0m
- KUKA IIWA: The base KUKA model has no gripper. We CREATE a visual parallel-jaw
  gripper and attach it to the end-effector using a constraint.

VISUAL GRIPPER FOR KUKA:
------------------------
Since the KUKA IIWA URDF doesn't include a gripper, we programmatically create
a simple parallel-jaw gripper using PyBullet shapes. This gripper:
- Has two finger "pads" that move in/out
- Is attached to the robot's end-effector via a fixed constraint
- Provides visual feedback of grasping actions

HOME POSE:
----------
A neutral "home" position is defined for each robot type so the robot
can return to a known safe configuration after completing repair tasks.
"""

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

# Global reference to the visual gripper (for KUKA)
_visual_gripper = None
_gripper_constraint = None


def get_visual_gripper_body():
    """Get the visual gripper body ID (for collision checking).
    
    Returns:
        int or None: The PyBullet body ID of the visual gripper, or None
                     if no visual gripper exists (e.g., when using Panda).
    """
    if _visual_gripper is not None:
        return _visual_gripper.get('body_id')
    return None


# ============================================================================
# TCP (TOOL CENTER POINT) OFFSET
# ============================================================================
# The KUKA ee_link (link 6) is the flange - but the visual gripper extends
# BELOW it. We need to plan to the TCP (the point between the fingertips)
# rather than the flange.
#
# Gripper geometry:
#   palm_size[2] = 0.04m (palm thickness in Z)
#   finger_size[2] = 0.12m (finger length in Z, extending down from palm)
#   Palm center is at ee_link, fingers extend -Z from palm bottom
#   TCP is at fingertip level = -(palm/2 + finger_length) = -(0.02 + 0.12) = -0.14m
#
# When we want the TCP at position P, the ee_link (flange) must be at P + [0, 0, +0.14]

KUKA_TCP_OFFSET_LOCAL = [0.0, 0.0, -0.14]  # TCP position in ee_link frame (fingertips below flange)


def tcp_to_ee_target(tcp_world_pos, ee_world_orn, tcp_offset_local=None):
    """Convert a desired TCP world position to the required ee_link (flange) position.
    
    When planning, we want the FINGERTIPS (TCP) to reach a certain point.
    But IK solves for the ee_link (flange). This function computes where
    the flange needs to be so that the TCP ends up at the desired location.
    
    Args:
        tcp_world_pos: Desired TCP position in world frame [x, y, z]
        ee_world_orn: Desired ee_link orientation as quaternion [x, y, z, w]
                      (the TCP has the same orientation as ee_link)
        tcp_offset_local: TCP offset in ee_link local frame [x, y, z].
                          Default uses KUKA_TCP_OFFSET_LOCAL = [0, 0, -0.14]
    
    Returns:
        ee_world_pos: The ee_link position that puts TCP at tcp_world_pos
    
    Math:
        tcp_world = ee_world + R_ee * tcp_local
        => ee_world = tcp_world - R_ee * tcp_local
        
    Where R_ee is the rotation matrix of the ee_link.
    """
    import numpy as np
    
    if tcp_offset_local is None:
        tcp_offset_local = KUKA_TCP_OFFSET_LOCAL
    
    tcp_offset_local = np.array(tcp_offset_local)
    tcp_world_pos = np.array(tcp_world_pos)
    
    # Get the rotation matrix from the quaternion
    # PyBullet quaternion is [x, y, z, w]
    rot_matrix = np.array(p.getMatrixFromQuaternion(ee_world_orn)).reshape(3, 3)
    
    # Transform TCP offset from local to world frame
    tcp_offset_world = rot_matrix @ tcp_offset_local
    
    # ee_world = tcp_world - tcp_offset_world
    ee_world_pos = tcp_world_pos - tcp_offset_world
    
    return list(ee_world_pos)


def ee_to_tcp_pos(ee_world_pos, ee_world_orn, tcp_offset_local=None):
    """Convert current ee_link position to TCP position (inverse of tcp_to_ee_target).
    
    Args:
        ee_world_pos: Current ee_link position in world frame [x, y, z]
        ee_world_orn: Current ee_link orientation as quaternion [x, y, z, w]
        tcp_offset_local: TCP offset in ee_link local frame. Default uses KUKA_TCP_OFFSET_LOCAL.
    
    Returns:
        tcp_world_pos: The TCP position in world frame
    """
    import numpy as np
    
    if tcp_offset_local is None:
        tcp_offset_local = KUKA_TCP_OFFSET_LOCAL
    
    tcp_offset_local = np.array(tcp_offset_local)
    ee_world_pos = np.array(ee_world_pos)
    
    # Get the rotation matrix from the quaternion
    rot_matrix = np.array(p.getMatrixFromQuaternion(ee_world_orn)).reshape(3, 3)
    
    # Transform TCP offset from local to world frame
    tcp_offset_world = rot_matrix @ tcp_offset_local
    
    # tcp_world = ee_world + tcp_offset_world
    tcp_world_pos = ee_world_pos + tcp_offset_world
    
    return list(tcp_world_pos)


# Home pose joint configurations for supported robots
# These are safe neutral positions that avoid collisions with the workspace
HOME_POSES = {
    # Panda: 7 arm joints + 2 gripper joints (gripper open)
    "panda": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    # KUKA IIWA: 7 arm joints - upright position pointing away from chair
    # Joint order: A1 (base), A2, A3, A4, A5, A6, A7 (wrist)
    # This pose:
    #   - A1=0: base pointing forward (+X)
    #   - A2=-0.4: shoulder slightly back (away from chair)
    #   - A3=0: no rotation
    #   - A4=-1.5: elbow bent up (keeps arm high)
    #   - A5=0: no rotation
    #   - A6=1.2: wrist angled to keep gripper clear
    #   - A7=0: no rotation
    # This keeps the entire arm behind the robot base (negative X side)
    # and above the chair to avoid any collision with seat/legs.
    "kuka": [0.0, -0.4, 0.0, -1.5, 0.0, 1.2, 0.0]
}


def get_finger_tcp(gripper_body):
    """Get the TRUE Tool Center Point (TCP) = midpoint between finger tips.
    
    This is the REAL grasp point, not the flange or gripper base.
    All approach and grasp planning should use this point.
    
    Args:
        gripper_body: PyBullet body ID of the visual gripper
        
    Returns:
        tcp_pos: World position of the TCP (midpoint between fingertips)
        tcp_orn: Orientation (same as gripper base)
    """
    import numpy as np
    
    if gripper_body is None:
        return None, None
    
    # Get finger link states (links 0 and 1 are the finger links)
    left_finger_state = p.getLinkState(gripper_body, 0)
    right_finger_state = p.getLinkState(gripper_body, 1)
    
    left_tip = np.array(left_finger_state[0])
    right_tip = np.array(right_finger_state[0])
    
    # Finger tips are at the BOTTOM of the finger blocks
    # Finger size Z is 0.12m, link position is at center, so tip is 0.06m below
    # Account for gripper orientation (fingers may not point straight down)
    gripper_base_state = p.getBasePositionAndOrientation(gripper_body)
    gripper_orn = gripper_base_state[1]
    
    # The fingers extend in the local -Z direction of the gripper
    rot_matrix = np.array(p.getMatrixFromQuaternion(gripper_orn)).reshape(3, 3)
    finger_extension = rot_matrix @ np.array([0, 0, -0.06])  # 6cm below link center = tip
    
    left_tip_world = left_tip + finger_extension
    right_tip_world = right_tip + finger_extension
    
    # TCP = midpoint between fingertips
    tcp_pos = (left_tip_world + right_tip_world) / 2.0
    
    return list(tcp_pos), gripper_orn


def get_tcp_offset_from_ee(robot_id, ee_link, gripper_body):
    """Compute the current TCP offset from the EE link in world frame.
    
    This is the vector from EE link to the actual finger TCP.
    Use this to convert between EE commands and finger tip positions.
    
    Returns:
        offset_world: Vector from EE to TCP in world frame [x, y, z]
    """
    import numpy as np
    
    if gripper_body is None:
        return [0, 0, 0]
    
    # Get current EE position
    ee_state = p.getLinkState(robot_id, ee_link)
    ee_pos = np.array(ee_state[0])
    
    # Get actual TCP position
    tcp_pos, _ = get_finger_tcp(gripper_body)
    if tcp_pos is None:
        return [0, 0, 0]
    
    tcp_pos = np.array(tcp_pos)
    
    return list(tcp_pos - ee_pos)


def create_visual_gripper():
    """Create a visual parallel-jaw gripper as a separate PyBullet multi-body.
    
    This creates a gripper with:
    - A base/palm piece (attaches to robot)
    - Two finger pads that slide in/out on prismatic joints
    
    The gripper is oriented so that:
    - The palm attaches to the robot's tool flange
    - The fingers extend downward (in -Z direction in gripper local frame)
    - The fingers open/close along the X axis
    
    Returns:
        tuple: (gripper_body_id, [left_finger_joint, right_finger_joint], open_val, close_val)
        
    The gripper dimensions are designed to be visible and proportional to chair parts.
    """
    # Gripper dimensions (meters) - industrial-scale parallel jaw gripper
    # Designed to realistically grasp chair parts (5-10cm thick)
    palm_size = [0.12, 0.08, 0.04]  # Palm/base: wider for stability
    finger_size = [0.025, 0.06, 0.12]  # Fingers: thicker, longer for secure grip
    
    # Colors - metallic industrial look
    palm_color = [0.3, 0.3, 0.35, 1.0]  # Steel gray
    finger_color = [0.15, 0.15, 0.2, 1.0]  # Dark metallic
    
    # Create collision and visual shapes for palm
    palm_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in palm_size])
    palm_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in palm_size], 
                                    rgbaColor=palm_color)
    
    # Create collision and visual shapes for fingers
    finger_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in finger_size])
    finger_vis_left = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in finger_size],
                                           rgbaColor=finger_color)
    finger_vis_right = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in finger_size],
                                            rgbaColor=finger_color)
    
    # Finger positions relative to palm center
    # Fingers extend downward (-Z) from the palm
    # They are spread apart along the X axis
    finger_x_offset = 0.06  # How far apart fingers are when open - wider spread
    finger_z_offset = -(palm_size[2]/2 + finger_size[2]/2)  # Below the palm
    
    # Create the multi-body gripper
    # Link 0: Left finger (prismatic, moves in +X to close)
    # Link 1: Right finger (prismatic, moves in -X to close)
    # ARCHITECTURAL FIX: Use realistic mass for stable contact physics
    gripper_id = p.createMultiBody(
        baseMass=0.3,  # Realistic palm mass (was 0.01)
        baseCollisionShapeIndex=palm_col,
        baseVisualShapeIndex=palm_vis,
        basePosition=[0, 0, 1],  # Will be repositioned when attached
        baseOrientation=[0, 0, 0, 1],
        linkMasses=[0.08, 0.08],  # Realistic finger mass (was 0.005)
        linkCollisionShapeIndices=[finger_col, finger_col],
        linkVisualShapeIndices=[finger_vis_left, finger_vis_right],
        linkPositions=[
            [-finger_x_offset, 0, finger_z_offset],  # Left finger (negative X)
            [finger_x_offset, 0, finger_z_offset]    # Right finger (positive X)
        ],
        linkOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],
        linkInertialFramePositions=[
            [0, 0, 0],
            [0, 0, 0]
        ],
        linkInertialFrameOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],
        linkParentIndices=[0, 0],  # Both fingers attached to base
        linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC],
        linkJointAxis=[
            [1, 0, 0],   # Left finger moves in +X (inward to close)
            [-1, 0, 0]   # Right finger moves in -X (inward to close)
        ]
    )
    
    # =========================================================================
    # GRIPPER PHYSICS SETTINGS (ARCHITECTURAL FIX)
    # =========================================================================
    # Set proper dynamics on gripper base and fingers for stable contact grasping.
    # Without these settings, objects slip through fingers or jitter during motion.
    
    # Gripper base (palm) dynamics
    p.changeDynamics(
        gripper_id, -1,  # -1 = base link
        lateralFriction=1.5,       # High friction for grip
        spinningFriction=0.3,      # Resist spinning in grip
        rollingFriction=0.2,       # Resist rolling
        restitution=0.0,           # No bounce
        linearDamping=0.1,         # Damping for stability
        angularDamping=0.1
    )
    
    # Finger dynamics - same settings for both fingers
    for joint_idx in [0, 1]:
        p.changeDynamics(
            gripper_id, joint_idx,
            lateralFriction=2.0,       # Very high friction on fingertips
            spinningFriction=0.5,      # Strong spin resistance
            rollingFriction=0.3,       # Strong roll resistance
            restitution=0.0,           # No bounce
            linearDamping=0.15,        # More damping on fingers
            angularDamping=0.15,
            jointDamping=0.5           # Joint damping for smooth motion
        )
        # Initialize to open position
        p.resetJointState(gripper_id, joint_idx, 0.0)
    
    # Set joint motor parameters for stronger grip force
    open_val = 0.0
    close_val = 0.05
    
    print(f"[GRIPPER] Created visual parallel-jaw gripper (body_id={gripper_id})")
    print(f"[GRIPPER]   Finger joints: [0, 1], open={open_val}, close={close_val}")
    print(f"[GRIPPER]   Physics: high friction, damping, realistic mass")
    
    return gripper_id, [0, 1], open_val, close_val


def attach_gripper_to_robot(robot_id, ee_link, gripper_id):
    """Attach the visual gripper to the robot's end-effector using a constraint.
    
    The gripper is oriented so that:
    - The palm is flush with the robot's tool flange
    - The fingers point in the direction the EE is facing (typically downward)
    
    Args:
        robot_id: PyBullet body ID of the robot
        ee_link: End-effector link index
        gripper_id: PyBullet body ID of the gripper
        
    Returns:
        int: Constraint ID
    """
    global _gripper_constraint
    
    # Get current end-effector pose
    ee_state = p.getLinkState(robot_id, ee_link)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    # Calculate the gripper orientation in world frame
    # We want to rotate the gripper 180 degrees around the EE's local X axis
    # so that the fingers point outward from the flange
    local_rotation = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    # Combine EE orientation with local rotation
    gripper_world_orn = p.multiplyTransforms(
        [0, 0, 0], ee_orn,
        [0, 0, 0], local_rotation
    )[1]
    
    # First, teleport the gripper to the EE location with correct orientation
    p.resetBasePositionAndOrientation(gripper_id, ee_pos, gripper_world_orn)
    
    # Step simulation briefly to let physics settle
    for _ in range(10):
        p.stepSimulation()
    
    # Create fixed constraint to attach gripper to robot EE
    # The constraint keeps the gripper attached as the robot moves
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=gripper_id,
        childLinkIndex=-1,  # Base of gripper
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],  # Attach at EE origin
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=local_rotation,  # Rotate gripper relative to EE
        childFrameOrientation=[0, 0, 0, 1]
    )
    
    # Make constraint very stiff so gripper follows EE exactly
    p.changeConstraint(constraint_id, maxForce=100000)
    
    # Step again to apply constraint
    for _ in range(10):
        p.stepSimulation()
    
    # Disable collisions between gripper and robot
    # This prevents the gripper from blocking robot motion
    num_robot_links = p.getNumJoints(robot_id)
    num_gripper_links = p.getNumJoints(gripper_id)
    
    # Disable collision between gripper base and all robot links
    for robot_link in range(-1, num_robot_links):
        p.setCollisionFilterPair(robot_id, gripper_id, robot_link, -1, enableCollision=0)
    
    # Disable collision between all gripper links and all robot links
    for gripper_link in range(num_gripper_links):
        for robot_link in range(-1, num_robot_links):
            p.setCollisionFilterPair(robot_id, gripper_id, robot_link, gripper_link, enableCollision=0)
    
    _gripper_constraint = constraint_id
    
    print(f"[GRIPPER] Attached gripper to robot EE (constraint_id={constraint_id})")
    
    return constraint_id


def discover_gripper_joints(robot_id, robot_type):
    """Discover and document gripper joint indices for a robot.
    
    This function inspects the robot URDF to find gripper-related joints
    by checking joint names for keywords like 'finger', 'gripper', 'hand'.
    
    Args:
        robot_id: PyBullet body ID of the robot
        robot_type: String identifier ("panda" or "kuka")
        
    Returns:
        tuple: (gripper_joint_indices, open_value, close_value)
        
    Notes:
        - Panda gripper joints are prismatic with range [0, 0.04]
        - KUKA has no built-in gripper; returns empty list (uses visual gripper)
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
        print(f"[GRIPPER]   No physical gripper joints found - will create visual gripper")
        return [], 0.0, 0.025


def load_robot(robot: str = "kuka"):
    """Load a robot URDF and return helper info.

    Args:
        robot: Identifier string. Supported values: "panda" (Franka Panda)
               or any other value (defaults to a KUKA IIWA model).

    Returns:
        A tuple: (robot_id, ee_link_index, gripper_joint_list, open_val, close_val)

    Notes:
        - For the Panda robot we expose gripper joints and open/close values
          so the executor can operate the gripper.
        - For KUKA, we CREATE a visual gripper and attach it to the end-effector.
        - The gripper info returned refers to the visual gripper for KUKA.
    """
    global _visual_gripper
    
    if robot == "panda":
        rid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 11  # panda_hand link (end-effector)
        
        # Discover gripper joints dynamically
        gripper, open_val, close_val = discover_gripper_joints(rid, "panda")
        
        # Fallback to known Panda joint indices if discovery fails
        if not gripper:
            gripper = [9, 10]  # panda_finger_joint1, panda_finger_joint2
            open_val, close_val = 0.04, 0.0
            print(f"[GRIPPER] Using fallback Panda gripper joints: {gripper}")
            
    else:
        # KUKA IIWA - no built-in gripper
        rid = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 6  # End-effector link (tool flange)
        
        # Create and attach a visual gripper
        gripper_id, gripper_joints, open_val, close_val = create_visual_gripper()
        attach_gripper_to_robot(rid, ee_link, gripper_id)
        
        # Store the visual gripper info globally for open/close operations
        _visual_gripper = {
            'body_id': gripper_id,
            'joints': gripper_joints,
            'open_val': open_val,
            'close_val': close_val
        }
        
        # Return the visual gripper's joints (these are on a separate body)
        # We'll handle this specially in open/close functions
        gripper = gripper_joints  # [0, 1] for left and right fingers
        
    # Log the robot configuration
    num_joints = p.getNumJoints(rid)
    print(f"[ROBOT] Loaded {robot} robot: id={rid}, ee_link={ee_link}, "
          f"joints={num_joints}, gripper_joints={gripper}")
            
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
    """Set gripper joints to `val` to open the gripper and step the sim.
    
    Also releases any active grasp constraint so the held object is freed.
    
    For KUKA robots with a visual gripper, this operates the separate
    gripper body instead of joints on the robot itself.
    
    Args:
        robot: PyBullet body ID of the robot
        joints: List of gripper joint indices
        val: Target position for gripper joints (open position)
    """
    global _active_grasp_constraint, _visual_gripper
    
    # First, release any active grasp constraint
    if _active_grasp_constraint is not None:
        try:
            p.removeConstraint(_active_grasp_constraint)
            print(f"[GRIPPER] Released grasp constraint {_active_grasp_constraint}")
        except Exception as e:
            print(f"[GRIPPER] Warning: Could not remove constraint: {e}")
        _active_grasp_constraint = None
    
    # Check if we're using a visual gripper (KUKA case)
    if _visual_gripper is not None:
        gripper_body = _visual_gripper['body_id']
        gripper_joints = _visual_gripper['joints']
        open_val = _visual_gripper['open_val']
        
        print(f"[GRIPPER] Opening visual gripper (fingers to position {open_val})")
        for j in gripper_joints:
            try:
                p.setJointMotorControl2(
                    gripper_body, j, p.POSITION_CONTROL,
                    open_val,
                    force=100,
                    maxVelocity=1.0
                )
            except Exception as e:
                print(f"[WARNING] Could not open visual gripper joint {j}: {e}")
        step_sim(0.3)
        return
    
    # Otherwise, use physical gripper joints on the robot
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
        step_sim(0.3)  # Give time for gripper to open
    else:
        # No gripper at all - just step simulation
        step_sim(0.1)


def close_gripper(robot, joints, val):
    """Set gripper joints to `val` to close the gripper and step the sim.
    
    This closes the physical or visual gripper joints. Actual grasping of objects
    is handled separately via create_grasp_constraint().
    
    For KUKA robots with a visual gripper, this operates the separate
    gripper body instead of joints on the robot itself.
    
    Args:
        robot: PyBullet body ID of the robot
        joints: List of gripper joint indices
        val: Target position for gripper joints (closed position)
    """
    global _visual_gripper
    
    # Check if we're using a visual gripper (KUKA case)
    if _visual_gripper is not None:
        gripper_body = _visual_gripper['body_id']
        gripper_joints = _visual_gripper['joints']
        close_val = _visual_gripper['close_val']
        
        print(f"[GRIPPER] Closing visual gripper (fingers to position {close_val})")
        for j in gripper_joints:
            try:
                p.setJointMotorControl2(
                    gripper_body, j, p.POSITION_CONTROL,
                    close_val,
                    force=150,
                    maxVelocity=1.0
                )
            except Exception as e:
                print(f"[WARNING] Could not close visual gripper joint {j}: {e}")
        step_sim(0.3)
        return
    
    # Otherwise, use physical gripper joints on the robot
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
        step_sim(0.3)  # Give time for gripper to close
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
    """
    Create a fixed constraint (grasp) between the gripper and target object.

    FIX:
    - Use closest-point pivots between gripper and object so the object is anchored
      where contact/proximity actually is (between/near the jaws), instead of welding
      the object's origin to an arbitrary EE frame.
    """
    global _active_grasp_constraint, _visual_gripper

    # First, release any existing grasp
    if _active_grasp_constraint is not None:
        try:
            p.removeConstraint(_active_grasp_constraint)
        except:
            pass
        _active_grasp_constraint = None

    try:
        # 1) Attach to the visual gripper if present (it is constrained to EE already)
        # 2) Build constraint using closest point pair -> correct visual anchoring
        parent_body = (_visual_gripper['body_id'] if _visual_gripper is not None else robot)
        parent_link = (-1 if _visual_gripper is not None else ee_link)

        # Collect closest-point candidates
        candidates = []
        try:
            candidates += list(p.getClosestPoints(parent_body, target_body, distance=0.20, linkIndexA=parent_link))
        except Exception:
            pass

        # If gripper is a multibody, also sample finger links
        if _visual_gripper is not None:
            try:
                g_id = _visual_gripper['body_id']
                for finger_link in range(p.getNumJoints(g_id)):
                    candidates += list(p.getClosestPoints(g_id, target_body, distance=0.20, linkIndexA=finger_link))
            except Exception:
                pass

        if not candidates:
            raise RuntimeError("No closest points returned between gripper and target")

        # Best = minimum distance
        best = min(candidates, key=lambda c: c[8])  # contactDistance
        dist = float(best[8])
        pos_on_a = best[5]  # world position on gripper/EE
        pos_on_b = best[6]  # world position on object

        # Parent pose
        if parent_link == -1:
            parent_pos, parent_orn = p.getBasePositionAndOrientation(parent_body)
        else:
            ls = p.getLinkState(parent_body, parent_link)
            parent_pos, parent_orn = ls[0], ls[1]

        # Child pose
        if target_link == -1:
            child_pos, child_orn = p.getBasePositionAndOrientation(target_body)
        else:
            ls = p.getLinkState(target_body, target_link)
            child_pos, child_orn = ls[0], ls[1]

        inv_parent_pos, inv_parent_orn = p.invertTransform(parent_pos, parent_orn)
        inv_child_pos, inv_child_orn = p.invertTransform(child_pos, child_orn)

        parent_pivot_local, _ = p.multiplyTransforms(inv_parent_pos, inv_parent_orn, pos_on_a, [0, 0, 0, 1])
        child_pivot_local, _ = p.multiplyTransforms(inv_child_pos, inv_child_orn, pos_on_b, [0, 0, 0, 1])

        attach_label = "VISUAL GRIPPER" if _visual_gripper is not None else f"robot EE link {ee_link}"
        print(f"[GRIPPER] Attaching object {target_body} to {attach_label} using closest-point pivots")
        print(f"[GRIPPER] Closest distance = {dist*1000:.1f}mm")

        constraint_id = p.createConstraint(
            parentBodyUniqueId=parent_body,
            parentLinkIndex=parent_link,
            childBodyUniqueId=target_body,
            childLinkIndex=target_link,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=parent_pivot_local,
            childFramePosition=child_pivot_local,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1],
        )

        p.changeConstraint(constraint_id, maxForce=100000)

        _active_grasp_constraint = constraint_id
        print(f"[GRIPPER] Created grasp constraint {constraint_id} for body {target_body}")
        return constraint_id

    except Exception as e:
        print(f"[GRIPPER] Failed to create grasp constraint: {e}")
        import traceback
        traceback.print_exc()
        return None


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


def move_to_home(robot, robot_type="kuka"):
    """Move the robot to its home/neutral position.
    
    This function commands all joints to their home configuration,
    providing a safe reset position after completing tasks.
    
    Args:
        robot: PyBullet body ID of the robot
        robot_type: Robot identifier ("panda" or "kuka")
        
    Notes:
        - Home pose is defined to avoid collisions with typical workspace
        - For Panda, gripper is opened in home pose
        - Motion uses position control with moderate speed
    """
    global _active_grasp_constraint
    
    # First, release any grasped object
    if _active_grasp_constraint is not None:
        release_grasp()
    
    # Get the home pose for this robot type
    home_pose = HOME_POSES.get(robot_type, HOME_POSES["kuka"])
    
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
