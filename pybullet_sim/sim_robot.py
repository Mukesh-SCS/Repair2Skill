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


# Home pose joint configurations for supported robots
# These are safe neutral positions that avoid collisions with the workspace
HOME_POSES = {
    # Panda: 7 arm joints + 2 gripper joints (gripper open)
    "panda": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    # KUKA IIWA: 7 arm joints in a neutral upright position
    "kuka": [0.0, 0.5, 0.0, -1.4, 0.0, 1.2, 0.0]
}


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
    # Using very low mass to minimize effect on robot dynamics
    gripper_id = p.createMultiBody(
        baseMass=0.01,  # Very light - just for visual
        baseCollisionShapeIndex=palm_col,
        baseVisualShapeIndex=palm_vis,
        basePosition=[0, 0, 1],  # Will be repositioned when attached
        baseOrientation=[0, 0, 0, 1],
        linkMasses=[0.005, 0.005],  # Light fingers
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
    
    # Set joint dynamics (NOTE: jointLowerLimit/jointUpperLimit are NOT valid
    # for changeDynamics - limits are set at creation time. We just set damping.)
    # Open position = 0 (fingers at default spread)
    # Close position = 0.05 (fingers moved inward) - larger travel for bigger parts
    open_val = 0.0
    close_val = 0.05
    
    for joint_idx in [0, 1]:
        p.changeDynamics(gripper_id, joint_idx, jointDamping=0.1)
        # Initialize to open position
        p.resetJointState(gripper_id, joint_idx, open_val)
    
    print(f"[GRIPPER] Created visual parallel-jaw gripper (body_id={gripper_id})")
    print(f"[GRIPPER]   Finger joints: [0, 1], open={open_val}, close={close_val}")
    
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
                    force=20,
                    maxVelocity=0.3
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
                    force=30,
                    maxVelocity=0.3
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


def create_grasp_constraint(robot, ee_link, target_body, target_link=-1):
    """Create a fixed constraint to attach an object to the robot's end-effector.
    
    This is the core of our constraint-based grasping system. Instead of
    relying on friction (which can be unreliable), we create a rigid
    constraint that "welds" the object to the end-effector/gripper.
    
    IMPORTANT: The constraint uses proper transform math to compute the
    child's position in the parent (EE) frame using inverse transforms.
    This ensures the object attaches exactly where it is relative to the
    gripper, preventing snaps, drifting, and unrealistic grabbing.
    
    Args:
        robot: PyBullet body ID of the robot
        ee_link: End-effector link index
        target_body: Body ID of the object to grasp
        target_link: Link index of the object (-1 for base link)
        
    Returns:
        int: Constraint ID if successful, None otherwise
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
        # Get parent (EE) pose in world frame
        ee_state = p.getLinkState(robot, ee_link)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        print(f"[GRIPPER] Attaching object {target_body} to robot EE link {ee_link}")
        
        # Get child (object) pose in world frame
        if target_link == -1:
            obj_pos, obj_orn = p.getBasePositionAndOrientation(target_body)
        else:
            link_state = p.getLinkState(target_body, target_link)
            obj_pos, obj_orn = link_state[0], link_state[1]
        
        # ===================================================================
        # CORRECT TRANSFORM MATH:
        # Compute child transform in parent (EE) local frame using:
        #   T_child_in_parent = T_parent^-1 * T_child_world
        # 
        # This is the PROPER way to set parentFramePosition/Orientation.
        # The old code used world-space offset which caused snapping/drifting.
        # ===================================================================
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        child_pos_in_ee, child_orn_in_ee = p.multiplyTransforms(
            inv_ee_pos, inv_ee_orn,
            obj_pos, obj_orn
        )
        
        print(f"[GRIPPER] Object at {[round(x,3) for x in obj_pos]}, EE at {[round(x,3) for x in ee_pos]}")
        print(f"[GRIPPER] Child in EE frame: pos={[round(x,3) for x in child_pos_in_ee]}")
        
        # Create fixed constraint between robot EE and object
        # parentFramePosition: position of child in parent's local frame
        # childFramePosition: [0,0,0] - anchor at child's origin
        # parentFrameOrientation: orientation of child in parent's local frame
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee_link,
            childBodyUniqueId=target_body,
            childLinkIndex=target_link,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=child_pos_in_ee,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=child_orn_in_ee,
            childFrameOrientation=[0, 0, 0, 1]
        )
        
        # Set realistic constraint force (5000-20000 range is good)
        # - Small parts: 5000-8000
        # - Medium parts (armrests): 8000-12000
        # - Large parts (seat, backrest): 10000-15000
        # Too high can cause instability, too low causes drift/lag
        p.changeConstraint(constraint_id, maxForce=12000)
        
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
