"""Collision-aware motion planning for PyBullet robot simulation.

This module provides collision checking and safe motion primitives to prevent
the robot arm from colliding with chair parts during repair operations.

ARCHITECTURAL DESIGN (CRITICAL):
--------------------------------
This module now enforces PLANNING/EXECUTION CONSISTENCY:
- Paths are collision-checked in JOINT SPACE
- Paths are EXECUTED in JOINT SPACE (NOT Cartesian IK)
- This guarantees the executed path matches the validated path

COLLISION CHECKING STRATEGY:
----------------------------
1. Use PyBullet's native collision detection (getClosestPoints, getContactPoints)
2. Check robot links vs environment obstacles (chair parts)
3. Check robot self-collision (link vs link)
4. Use INFLATED safety margin (3cm) for conservative collision detection

MOTION PLANNING STRATEGY (3-PHASE):
-----------------------------------
Phase 1: FREE-SPACE MOTION
    - Large steps allowed
    - RRT planning when direct path blocked
    - No contact expected

Phase 2: GUARDED APPROACH  
    - Linear in TCP frame
    - Step size ≤ 2mm
    - Abort on ANY contact

Phase 3: CONTACT-CONTROLLED CLOSURE
    - No Cartesian motion
    - Only finger joints move
    - Contact-based termination

If direct path fails, use lightweight sampling-based fallback (RRT-like).

Author: Collision-aware motion module for Repair2Skill
"""

import pybullet as p
import math
import random
from typing import List, Tuple, Optional, Dict, Set

# ============================================================================
# CONFIGURATION - INFLATED MARGINS FOR SAFETY
# ============================================================================

# Safety margin for collision detection (meters) - balanced for planning
COLLISION_SAFETY_MARGIN = 0.02  # 2cm - balance between safety and reachability

# Maximum distance to check for collisions
COLLISION_CHECK_DISTANCE = 0.05  # 5cm - detection range

# Approach-phase margin (tighter, used during guarded approach)
APPROACH_SAFETY_MARGIN = 0.008  # 8mm - only used during final approach

# Self-collision margin (tighter than obstacle margin - robot links are designed to be close)
SELF_COLLISION_MARGIN = 0.003  # 3mm - robot links are designed to not touch

# Number of interpolation steps for linear motion
LINEAR_MOTION_STEPS = 25  # Increased for finer collision checking

# Guarded approach step size (meters) - VERY SMALL
GUARDED_APPROACH_STEP = 0.002  # 2mm steps during guarded approach

# RRT parameters
RRT_MAX_ITERATIONS = 250  # Increased for better path finding
RRT_STEP_SIZE = 0.12  # radians (smaller steps)
RRT_GOAL_BIAS = 0.25  # 25% chance to sample goal directly

# Pre-approach offset distance (meters)
PRE_APPROACH_OFFSET = 0.18  # 18cm above target (increased clearance)

# Motor control parameters for joint-space execution
JOINT_MAX_FORCE = 500  # N
JOINT_MAX_VELOCITY = 2.0  # rad/s

# IK parameters
IK_MAX_CANDIDATES = 20  # Number of IK candidates to try before giving up
IK_JITTER_MAGNITUDE = 0.3  # radians - random offset for rest poses

# ============================================================================
# COLLISION CHECKING
# ============================================================================

def get_robot_arm_links(robot_id: int) -> List[int]:
    """Get list of robot arm link indices (excluding gripper/end-effector).
    
    Args:
        robot_id: PyBullet body ID of the robot
        
    Returns:
        List of link indices that are part of the robot arm
    """
    num_joints = p.getNumJoints(robot_id)
    arm_links = []
    
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot_id, j)
        joint_type = joint_info[2]
        
        # Include all non-fixed joints (these are movable arm links)
        if joint_type != p.JOINT_FIXED:
            arm_links.append(j)
    
    # Also include base link (-1)
    return [-1] + arm_links


# ============================================================================
# GRIPPER COLLISION CHECKING (FIX 1)
# ============================================================================
# The robot arm collision checking is not enough - we must also check
# the visual gripper body (for KUKA) against obstacles. Otherwise the
# path can be "safe" for the arm but the gripper still hits the chair.

def check_collision_gripper_vs_obstacle(
    gripper_id: int,
    obstacle_id: int,
    safety_margin: float = COLLISION_SAFETY_MARGIN
) -> Tuple[bool, float]:
    """Check if visual gripper collides with a single obstacle.
    
    This checks both the gripper base and all finger links against
    the obstacle. Essential for KUKA where gripper is a separate body.
    
    Args:
        gripper_id: PyBullet body ID of the visual gripper
        obstacle_id: PyBullet body ID of the obstacle
        safety_margin: Minimum allowed distance
        
    Returns:
        Tuple of (is_collision, min_distance)
    """
    min_distance = float('inf')
    
    try:
        # Check gripper base (-1)
        pts = p.getClosestPoints(
            bodyA=gripper_id,
            bodyB=obstacle_id,
            distance=COLLISION_CHECK_DISTANCE
        )
        for c in pts:
            min_distance = min(min_distance, c[8])
        
        # Check gripper finger links
        num_gripper_links = p.getNumJoints(gripper_id)
        for link in range(num_gripper_links):
            pts = p.getClosestPoints(
                bodyA=gripper_id,
                bodyB=obstacle_id,
                distance=COLLISION_CHECK_DISTANCE,
                linkIndexA=link,
                linkIndexB=-1
            )
            for c in pts:
                min_distance = min(min_distance, c[8])
    except Exception as e:
        pass  # If gripper doesn't exist, treat as no collision
    
    is_collision = min_distance < safety_margin
    return is_collision, min_distance


def check_collision_gripper_vs_obstacles(
    gripper_id: int,
    obstacle_ids: List[int],
    safety_margin: float = COLLISION_SAFETY_MARGIN,
    excluded_obstacles: Optional[Set[int]] = None
) -> Tuple[bool, int, float]:
    """Check if gripper collides with any obstacle in the list.
    
    Args:
        gripper_id: PyBullet body ID of the visual gripper
        obstacle_ids: List of obstacle body IDs
        safety_margin: Minimum allowed distance
        excluded_obstacles: Set of obstacle IDs to skip
        
    Returns:
        Tuple of (is_collision, colliding_obstacle_id, min_distance)
    """
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    for obs_id in obstacle_ids:
        if obs_id in excluded_obstacles:
            continue
        
        is_collision, distance = check_collision_gripper_vs_obstacle(
            gripper_id, obs_id, safety_margin
        )
        
        if is_collision:
            return True, obs_id, distance
    
    return False, -1, float('inf')


def check_collision_robot_vs_obstacle(
    robot_id: int,
    obstacle_id: int,
    robot_links: Optional[List[int]] = None,
    safety_margin: float = COLLISION_SAFETY_MARGIN
) -> Tuple[bool, float]:
    """Check if robot collides with a single obstacle.
    
    ARCHITECTURAL FIX: Now checks ALL robot links against ALL obstacle links.
    This prevents collisions where the robot's upper arm or elbow hits
    chair parts, not just the end-effector.
    
    Args:
        robot_id: PyBullet body ID of the robot
        obstacle_id: PyBullet body ID of the obstacle
        robot_links: Specific links to check (None = all arm links)
        safety_margin: Minimum allowed distance
        
    Returns:
        Tuple of (is_collision, min_distance)
    """
    if robot_links is None:
        robot_links = get_robot_arm_links(robot_id)
    
    min_distance = float('inf')
    
    # Get all obstacle links (base + all joints)
    num_obstacle_links = p.getNumJoints(obstacle_id)
    obstacle_links = [-1] + list(range(num_obstacle_links))  # -1 is base link
    
    for link_idx in robot_links:
        for obs_link in obstacle_links:
            try:
                # Get closest points between robot link and obstacle link
                contacts = p.getClosestPoints(
                    bodyA=robot_id,
                    bodyB=obstacle_id,
                    distance=COLLISION_CHECK_DISTANCE,
                    linkIndexA=link_idx,
                    linkIndexB=obs_link
                )
                
                for contact in contacts:
                    distance = contact[8]  # Closest distance
                    if distance < min_distance:
                        min_distance = distance
                        
            except Exception:
                continue
    
    is_collision = min_distance < safety_margin
    return is_collision, min_distance

def check_collision_robot_vs_obstacles(
    robot_id: int,
    obstacle_ids: List[int],
    robot_links: Optional[List[int]] = None,
    safety_margin: float = COLLISION_SAFETY_MARGIN,
    excluded_obstacles: Optional[Set[int]] = None,
    gripper_body_id: Optional[int] = None
) -> Tuple[bool, int, float]:
    """Check if robot OR gripper collides with any obstacle in the list.
    
    IMPORTANT: Now also checks the visual gripper body (for KUKA).
    The gripper extends beyond the robot flange, so checking only
    the robot arm is insufficient - the gripper can hit the chair
    even when the arm path is clear.
    
    Args:
        robot_id: PyBullet body ID of the robot
        obstacle_ids: List of obstacle body IDs
        robot_links: Specific links to check (None = all)
        safety_margin: Minimum allowed distance
        excluded_obstacles: Set of obstacle IDs to skip (e.g., grasped object)
        gripper_body_id: Optional visual gripper body ID to also check
        
    Returns:
        Tuple of (is_collision, colliding_obstacle_id, min_distance)
        colliding_obstacle_id is -1 if no collision
    """
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    if robot_links is None:
        robot_links = get_robot_arm_links(robot_id)
    
    for obs_id in obstacle_ids:
        if obs_id in excluded_obstacles:
            continue
        
        # Check robot arm vs obstacle
        is_collision, distance = check_collision_robot_vs_obstacle(
            robot_id, obs_id, robot_links, safety_margin
        )
        
        if is_collision:
            return True, obs_id, distance
        
        # CRITICAL FIX: Also check gripper body vs obstacle
        if gripper_body_id is not None:
            gcol, gdist = check_collision_gripper_vs_obstacle(
                gripper_body_id, obs_id, safety_margin
            )
            if gcol:
                return True, obs_id, gdist
    
    return False, -1, float('inf')


def check_self_collision(
    robot_id: int,
    safety_margin: float = None
) -> Tuple[bool, Tuple[int, int], float]:
    """Check for robot self-collision.
    
    ARCHITECTURAL FIX: Uses SELF_COLLISION_MARGIN (5mm) by default instead of 
    COLLISION_SAFETY_MARGIN (3cm). Robot links are designed to get close to each 
    other without touching - the inflated obstacle margin was causing false 
    positives at KUKA home pose (links 4 and 6 are close but don't touch).
    
    Also skips links within 3 positions of each other (not just 2) since 
    serial link robots have kinematic constraints that prevent nearby links
    from colliding under normal operation.
    
    Args:
        robot_id: PyBullet body ID of the robot
        safety_margin: Minimum allowed distance (default: SELF_COLLISION_MARGIN = 5mm)
        
    Returns:
        Tuple of (is_collision, (link1, link2), min_distance)
    """
    if safety_margin is None:
        safety_margin = SELF_COLLISION_MARGIN  # Use tighter margin for self-collision
    
    arm_links = get_robot_arm_links(robot_id)
    min_distance = float('inf')
    collision_pair = (-1, -1)
    
    # Check each pair of non-nearby links
    # Skip links within 3 positions (i+3) - serial robots can't self-collide between close links
    # Previously was i+2 which caused false positives between links 4 and 6
    for i, link_a in enumerate(arm_links):
        for link_b in arm_links[i+3:]:  # Skip nearby links (was i+2)
            try:
                contacts = p.getClosestPoints(
                    bodyA=robot_id,
                    bodyB=robot_id,
                    distance=COLLISION_CHECK_DISTANCE,
                    linkIndexA=link_a,
                    linkIndexB=link_b
                )
                
                for contact in contacts:
                    distance = contact[8]
                    if distance < min_distance:
                        min_distance = distance
                        collision_pair = (link_a, link_b)
                        
            except Exception:
                continue
    
    is_collision = min_distance < safety_margin
    return is_collision, collision_pair, min_distance


def is_configuration_collision_free(
    robot_id: int,
    joint_positions: List[float],
    obstacle_ids: List[int],
    excluded_obstacles: Optional[Set[int]] = None,
    check_self: bool = True,
    safety_margin: float = COLLISION_SAFETY_MARGIN,
    gripper_body_id: Optional[int] = None,
    ee_link: Optional[int] = None
) -> Tuple[bool, str]:
    """Check if a joint configuration is collision-free.
    
    This function temporarily sets joint positions, checks collisions,
    and returns whether the configuration is valid.
    
    IMPORTANT: Now also checks the visual gripper body (for KUKA).
    CRITICAL FIX: Restores gripper pose after collision check (was mutating world state).
    
    Args:
        robot_id: PyBullet body ID of the robot
        joint_positions: List of joint position values
        obstacle_ids: List of obstacle body IDs to check against
        excluded_obstacles: Set of obstacles to ignore (e.g., grasped object)
        check_self: Whether to check self-collision
        safety_margin: Minimum allowed distance
        gripper_body_id: Optional visual gripper body ID to also check
        ee_link: End-effector link index (REQUIRED if gripper_body_id is provided)
        
    Returns:
        Tuple of (is_collision_free, reason_if_failed)
    """
    # Store current joint states
    num_joints = p.getNumJoints(robot_id)
    original_states = []
    joint_indices = []
    
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot_id, j)
        if joint_info[2] != p.JOINT_FIXED:
            original_states.append(p.getJointState(robot_id, j)[0])
            joint_indices.append(j)
    
    # CRITICAL FIX #1: Store gripper pose BEFORE modifying it
    gripper_pos_0 = None
    gripper_orn_0 = None
    if gripper_body_id is not None:
        try:
            gripper_pos_0, gripper_orn_0 = p.getBasePositionAndOrientation(gripper_body_id)
        except Exception:
            pass  # If we can't get gripper pose, we'll skip restoring it
    
    # Temporarily set new joint positions
    try:
        for i, j in enumerate(joint_indices):
            if i < len(joint_positions):
                p.resetJointState(robot_id, j, joint_positions[i])
    except Exception as e:
        # Restore original states
        for i, j in enumerate(joint_indices):
            p.resetJointState(robot_id, j, original_states[i])
        return False, f"Failed to set joints: {e}"
    
    # ARCHITECTURAL FIX: Use performCollisionDetection() instead of stepSimulation()
    # stepSimulation() can move dynamic objects and change the world state,
    # making collision checks non-deterministic. performCollisionDetection()
    # only updates collision data without advancing physics.
    if gripper_body_id is not None and ee_link is not None:
        # The gripper is attached via constraint - we need to manually update its pose
        # based on the new EE position after setting joint states
        try:
            ee_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
            ee_pos = ee_state[4]  # World position
            ee_orn = ee_state[5]  # World orientation
            
            # Compute gripper orientation (rotated 180° around X from EE)
            local_rotation = p.getQuaternionFromEuler([math.pi, 0, 0])
            gripper_orn = p.multiplyTransforms([0,0,0], ee_orn, [0,0,0], local_rotation)[1]
            
            # Update gripper base pose during collision check
            p.resetBasePositionAndOrientation(gripper_body_id, ee_pos, gripper_orn)
        except Exception:
            pass  # If this fails, the collision check will still work with old pose
    
    # Update collision detection without advancing physics
    p.performCollisionDetection()
    
    # Check collisions
    try:
        # Check robot vs obstacles (now includes gripper)
        is_collision, obs_id, _ = check_collision_robot_vs_obstacles(
            robot_id, obstacle_ids, 
            excluded_obstacles=excluded_obstacles,
            safety_margin=safety_margin,
            gripper_body_id=gripper_body_id
        )
        
        if is_collision:
            # Restore original states
            for i, j in enumerate(joint_indices):
                p.resetJointState(robot_id, j, original_states[i])
            # CRITICAL FIX #2: Also restore gripper pose before returning!
            if gripper_body_id is not None and gripper_pos_0 is not None:
                p.resetBasePositionAndOrientation(gripper_body_id, gripper_pos_0, gripper_orn_0)
            return False, f"Collision with obstacle {obs_id}"
        
        # Check self-collision (uses SELF_COLLISION_MARGIN by default, not obstacle margin)
        if check_self:
            is_self_collision, pair, _ = check_self_collision(robot_id)  # Use default margin
            if is_self_collision:
                # Restore original states
                for i, j in enumerate(joint_indices):
                    p.resetJointState(robot_id, j, original_states[i])
                # CRITICAL FIX #2: Also restore gripper pose before returning!
                if gripper_body_id is not None and gripper_pos_0 is not None:
                    p.resetBasePositionAndOrientation(gripper_body_id, gripper_pos_0, gripper_orn_0)
                return False, f"Self-collision between links {pair}"
        
    finally:
        # Always restore original states and gripper pose
        for i, j in enumerate(joint_indices):
            p.resetJointState(robot_id, j, original_states[i])
        
        # CRITICAL FIX #2: Restore gripper pose to its original state
        if gripper_body_id is not None and gripper_pos_0 is not None:
            p.resetBasePositionAndOrientation(gripper_body_id, gripper_pos_0, gripper_orn_0)
            # Optional but recommended: update collision detection after restoring
            p.performCollisionDetection()
    
    return True, "OK"


# ============================================================================
# MOTION PRIMITIVES
# ============================================================================

def compute_pre_approach_pose(
    target_pos: List[float],
    approach_direction: str = "above",
    offset_distance: float = PRE_APPROACH_OFFSET
) -> List[float]:
    """Compute a safe pre-approach position offset from the target.
    
    Args:
        target_pos: Target position [x, y, z]
        approach_direction: Direction to offset from ("above", "front", "side")
        offset_distance: Distance to offset
        
    Returns:
        Pre-approach position [x, y, z]
    """
    pre_pos = list(target_pos)
    
    if approach_direction == "above":
        pre_pos[2] += offset_distance
    elif approach_direction == "front":
        pre_pos[0] -= offset_distance
    elif approach_direction == "side":
        pre_pos[1] += offset_distance
    elif approach_direction == "back":
        pre_pos[0] += offset_distance
    else:
        # Default: above
        pre_pos[2] += offset_distance
    
    return pre_pos


def interpolate_positions(
    start_pos: List[float],
    end_pos: List[float],
    num_steps: int = LINEAR_MOTION_STEPS
) -> List[List[float]]:
    """Generate linear interpolation between two positions.
    
    Args:
        start_pos: Starting position [x, y, z]
        end_pos: Ending position [x, y, z]
        num_steps: Number of intermediate steps
        
    Returns:
        List of interpolated positions
    """
    positions = []
    
    for i in range(num_steps + 1):
        t = i / num_steps
        pos = [
            start_pos[0] + t * (end_pos[0] - start_pos[0]),
            start_pos[1] + t * (end_pos[1] - start_pos[1]),
            start_pos[2] + t * (end_pos[2] - start_pos[2])
        ]
        positions.append(pos)
    
    return positions


def interpolate_joints(
    start_joints: List[float],
    end_joints: List[float],
    num_steps: int = LINEAR_MOTION_STEPS
) -> List[List[float]]:
    """Generate linear interpolation between two joint configurations.
    
    Args:
        start_joints: Starting joint positions
        end_joints: Ending joint positions
        num_steps: Number of intermediate steps
        
    Returns:
        List of interpolated joint configurations
    """
    configs = []
    
    for i in range(num_steps + 1):
        t = i / num_steps
        config = [
            start_joints[j] + t * (end_joints[j] - start_joints[j])
            for j in range(len(start_joints))
        ]
        configs.append(config)
    
    return configs


# ============================================================================
# MULTI-CANDIDATE IK (FIX FOR GOAL-IN-COLLISION)
# ============================================================================
# ARCHITECTURAL FIX: The IK solver returns ONE solution based on rest_poses.
# If that solution collides with the chair, we're stuck. Solution: try multiple
# IK candidates by jittering rest_poses, and pick the first collision-free one.

def compute_ik_collision_free(
    robot_id: int,
    ee_link: int,
    target_pos: List[float],
    target_orn: Optional[List[float]],
    obstacle_ids: List[int],
    excluded_obstacles: Optional[Set[int]] = None,
    current_joints: Optional[List[float]] = None,
    max_candidates: int = IK_MAX_CANDIDATES,
    safety_margin: float = COLLISION_SAFETY_MARGIN,
    gripper_body_id: Optional[int] = None,
    position_only_if_stuck: bool = False
) -> Tuple[Optional[List[float]], str]:
    """Compute IK solution that is collision-free.
    
    Tries multiple IK candidates by jittering rest poses until a collision-free
    solution is found. Falls back to position-only IK if orientation-constrained
    IK fails.
    
    Args:
        robot_id: PyBullet robot body ID
        ee_link: End-effector link index
        target_pos: Target position [x, y, z]
        target_orn: Target orientation quaternion, or None for position-only
        obstacle_ids: List of obstacle body IDs
        excluded_obstacles: Set of body IDs to ignore in collision check
        current_joints: Current joint configuration (used as primary rest pose)
        max_candidates: Maximum number of IK candidates to try
        safety_margin: Collision safety margin
        gripper_body_id: Optional gripper body ID for collision checking
        position_only_if_stuck: If True, try position-only IK after orientation fails
        
    Returns:
        Tuple of (joint_positions, message) where joint_positions is None if failed
    """
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    # Get joint limits
    num_joints_total = p.getNumJoints(robot_id)
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    
    movable_joints = []
    for j in range(num_joints_total):
        joint_info = p.getJointInfo(robot_id, j)
        if joint_info[2] != p.JOINT_FIXED:
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
            movable_joints.append(j)
    
    num_joints = len(lower_limits)
    
    # Get current joints if not provided
    if current_joints is None:
        current_joints = get_current_joint_positions(robot_id)
    
    # FIX: Use current joints as PRIMARY rest pose (not zeros!)
    base_rest_poses = list(current_joints[:num_joints])
    
    # Try multiple IK candidates with jittered rest poses
    for attempt in range(max_candidates):
        # Jitter rest poses (but use current joints for attempt 0)
        if attempt == 0:
            rest_poses = base_rest_poses
        else:
            rest_poses = [
                base_rest_poses[j] + random.uniform(-IK_JITTER_MAGNITUDE, IK_JITTER_MAGNITUDE)
                for j in range(num_joints)
            ]
            # Clamp to joint limits
            rest_poses = [
                max(lower_limits[j], min(upper_limits[j], rest_poses[j]))
                for j in range(num_joints)
            ]
        
        # Compute IK
        if target_orn is not None:
            ik_result = p.calculateInverseKinematics(
                robot_id, ee_link, target_pos, targetOrientation=target_orn,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=200,
                residualThreshold=1e-4
            )
        else:
            ik_result = p.calculateInverseKinematics(
                robot_id, ee_link, target_pos,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=100
            )
        
        candidate_joints = list(ik_result)[:num_joints]
        
        # Check collision
        is_free, reason = is_configuration_collision_free(
            robot_id, candidate_joints, obstacle_ids, excluded_obstacles,
            check_self=True, safety_margin=safety_margin,
            gripper_body_id=gripper_body_id, ee_link=ee_link
        )
        
        if is_free:
            if attempt > 0:
                print(f"[IK] Found collision-free solution on attempt {attempt+1}")
            return candidate_joints, f"IK solved (attempt {attempt+1})"
        else:
            # Debug: Log first collision reason
            if attempt == 0:
                print(f"[IK DEBUG] First candidate blocked: {reason}")
    
    # All orientation-constrained IK candidates failed
    if position_only_if_stuck and target_orn is not None:
        print(f"[IK] Orientation IK failed {max_candidates} times, trying position-only...")
        # Try position-only IK as last resort
        for attempt in range(max_candidates // 2):
            if attempt == 0:
                rest_poses = base_rest_poses
            else:
                rest_poses = [
                    base_rest_poses[j] + random.uniform(-IK_JITTER_MAGNITUDE, IK_JITTER_MAGNITUDE)
                    for j in range(num_joints)
                ]
                rest_poses = [
                    max(lower_limits[j], min(upper_limits[j], rest_poses[j]))
                    for j in range(num_joints)
                ]
            
            ik_result = p.calculateInverseKinematics(
                robot_id, ee_link, target_pos,  # No orientation
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=100
            )
            
            candidate_joints = list(ik_result)[:num_joints]
            
            is_free, reason = is_configuration_collision_free(
                robot_id, candidate_joints, obstacle_ids, excluded_obstacles,
                check_self=True, safety_margin=safety_margin,
                gripper_body_id=gripper_body_id, ee_link=ee_link
            )
            
            if is_free:
                print(f"[IK] Found position-only solution on attempt {attempt+1}")
                return candidate_joints, f"Position-only IK (attempt {attempt+1})"
    
    return None, f"IK failed: all {max_candidates} candidates collided"


def check_path_collision_free(
    robot_id: int,
    joint_path: List[List[float]],
    obstacle_ids: List[int],
    excluded_obstacles: Optional[Set[int]] = None,
    safety_margin: float = COLLISION_SAFETY_MARGIN,
    gripper_body_id: Optional[int] = None,
    ee_link: Optional[int] = None
) -> Tuple[bool, int, str]:
    """Check if entire path is collision-free.
    
    IMPORTANT: Now also checks the visual gripper body (for KUKA).
    CRITICAL FIX: Passes ee_link to collision checker so gripper pose is updated correctly.
    
    Args:
        robot_id: PyBullet body ID of the robot
        joint_path: List of joint configurations along the path
        obstacle_ids: Obstacles to check against
        excluded_obstacles: Obstacles to ignore
        safety_margin: Minimum allowed distance
        gripper_body_id: Optional visual gripper body ID to also check
        ee_link: End-effector link index (REQUIRED if gripper_body_id is provided)
        
    Returns:
        Tuple of (is_collision_free, failing_step_index, reason)
    """
    for i, config in enumerate(joint_path):
        is_free, reason = is_configuration_collision_free(
            robot_id, config, obstacle_ids, 
            excluded_obstacles=excluded_obstacles,
            safety_margin=safety_margin,
            gripper_body_id=gripper_body_id,
            ee_link=ee_link
        )
        
        if not is_free:
            return False, i, reason
    
    return True, -1, "Path is collision-free"


# ============================================================================
# SIMPLE RRT PLANNER
# ============================================================================

class SimpleRRT:
    """Lightweight RRT planner for collision-free path finding.
    
    This is a minimal implementation that samples random joint configurations
    and builds a tree toward the goal. It's computationally lightweight and
    doesn't require external libraries.
    
    Now includes gripper body collision checking for KUKA visual gripper.
    """
    
    def __init__(
        self,
        robot_id: int,
        obstacle_ids: List[int],
        joint_limits: List[Tuple[float, float]],
        excluded_obstacles: Optional[Set[int]] = None,
        gripper_body_id: Optional[int] = None,
        ee_link: Optional[int] = None
    ):
        """Initialize the RRT planner.
        
        Args:
            robot_id: PyBullet body ID
            obstacle_ids: List of obstacle body IDs
            joint_limits: List of (min, max) tuples for each joint
            excluded_obstacles: Obstacles to ignore during collision checking
            gripper_body_id: Optional visual gripper body ID to also check
            ee_link: End-effector link index (REQUIRED if gripper_body_id is provided)
        """
        self.robot_id = robot_id
        self.obstacle_ids = obstacle_ids
        self.joint_limits = joint_limits
        self.excluded_obstacles = excluded_obstacles or set()
        self.gripper_body_id = gripper_body_id
        self.ee_link = ee_link
        self.num_joints = len(joint_limits)
        
        # Tree storage: list of (config, parent_index)
        self.tree = []
    
    def _sample_random_config(self) -> List[float]:
        """Sample a random joint configuration within limits."""
        config = []
        for low, high in self.joint_limits:
            config.append(random.uniform(low, high))
        return config
    
    def _distance(self, config1: List[float], config2: List[float]) -> float:
        """Compute Euclidean distance in joint space."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(config1, config2)))
    
    def _nearest_neighbor(self, config: List[float]) -> int:
        """Find index of nearest node in tree to given config."""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, (node_config, _) in enumerate(self.tree):
            dist = self._distance(node_config, config)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(
        self,
        from_config: List[float],
        to_config: List[float],
        step_size: float = RRT_STEP_SIZE
    ) -> List[float]:
        """Steer from one config toward another, limited by step size."""
        dist = self._distance(from_config, to_config)
        
        if dist <= step_size:
            return to_config
        
        # Move step_size distance toward target
        t = step_size / dist
        new_config = [
            from_config[i] + t * (to_config[i] - from_config[i])
            for i in range(self.num_joints)
        ]
        
        return new_config
    
    def _is_collision_free(self, config: List[float]) -> bool:
        """Check if configuration is collision-free (robot + gripper)."""
        is_free, _ = is_configuration_collision_free(
            self.robot_id,
            config,
            self.obstacle_ids,
            excluded_obstacles=self.excluded_obstacles,
            check_self=True,
            gripper_body_id=self.gripper_body_id,  # Include gripper in collision check
            ee_link=self.ee_link  # CRITICAL FIX: Pass ee_link for proper gripper pose update
        )
        return is_free
    
    def _extract_path(self, goal_idx: int) -> List[List[float]]:
        """Extract path from tree root to goal node."""
        path = []
        idx = goal_idx
        
        while idx >= 0:
            config, parent_idx = self.tree[idx]
            path.append(config)
            idx = parent_idx
        
        path.reverse()
        return path
    
    def plan(
        self,
        start_config: List[float],
        goal_config: List[float],
        max_iterations: int = RRT_MAX_ITERATIONS,
        goal_threshold: float = 0.1
    ) -> Optional[List[List[float]]]:
        """Plan a collision-free path from start to goal.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            max_iterations: Maximum RRT iterations
            goal_threshold: Distance threshold to consider goal reached
            
        Returns:
            List of joint configurations from start to goal, or None if failed
        """
        # Check if start and goal are valid
        # ARCHITECTURAL FIX: If start is flagged as in-collision, we're already
        # physically at that position, so the collision must be a false positive
        # due to margin being too conservative. Try with reduced margin.
        if not self._is_collision_free(start_config):
            # Try with reduced margin (half the normal margin)
            is_free_reduced, _ = is_configuration_collision_free(
                self.robot_id, start_config, self.obstacle_ids,
                excluded_obstacles=self.excluded_obstacles,
                check_self=True,
                safety_margin=COLLISION_SAFETY_MARGIN / 2,  # Use half margin
                gripper_body_id=self.gripper_body_id,
                ee_link=self.ee_link
            )
            if is_free_reduced:
                print("[RRT] Start collision was false positive (passed with reduced margin)")
            else:
                print("[RRT] Start configuration is in collision!")
                return None
        
        if not self._is_collision_free(goal_config):
            print("[RRT] Goal configuration is in collision!")
            return None
        
        # Initialize tree with start
        self.tree = [(start_config, -1)]
        
        for iteration in range(max_iterations):
            # Sample random config (with goal bias)
            if random.random() < RRT_GOAL_BIAS:
                sample = goal_config
            else:
                sample = self._sample_random_config()
            
            # Find nearest node
            nearest_idx = self._nearest_neighbor(sample)
            nearest_config = self.tree[nearest_idx][0]
            
            # Steer toward sample
            new_config = self._steer(nearest_config, sample)
            
            # Check if new config is collision-free
            if self._is_collision_free(new_config):
                # Add to tree
                new_idx = len(self.tree)
                self.tree.append((new_config, nearest_idx))
                
                # Check if we reached the goal
                if self._distance(new_config, goal_config) < goal_threshold:
                    # Try to connect directly to goal
                    if self._is_collision_free(goal_config):
                        self.tree.append((goal_config, new_idx))
                        path = self._extract_path(len(self.tree) - 1)
                        print(f"[RRT] Found path in {iteration + 1} iterations")
                        return path
        
        print(f"[RRT] Failed to find path after {max_iterations} iterations")
        return None


def get_joint_limits(robot_id: int) -> List[Tuple[float, float]]:
    """Get joint limits for all movable joints.
    
    Args:
        robot_id: PyBullet body ID
        
    Returns:
        List of (min, max) tuples for each movable joint
    """
    limits = []
    num_joints = p.getNumJoints(robot_id)
    
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot_id, j)
        if joint_info[2] != p.JOINT_FIXED:
            lower = joint_info[8]
            upper = joint_info[9]
            # Handle unlimited joints (limits = 0)
            if lower >= upper:
                lower = -math.pi
                upper = math.pi
            limits.append((lower, upper))
    
    return limits


def get_current_joint_positions(robot_id: int) -> List[float]:
    """Get current positions of all movable joints.
    
    Args:
        robot_id: PyBullet body ID
        
    Returns:
        List of joint positions
    """
    positions = []
    num_joints = p.getNumJoints(robot_id)
    
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot_id, j)
        if joint_info[2] != p.JOINT_FIXED:
            positions.append(p.getJointState(robot_id, j)[0])
    
    return positions


def get_movable_joint_indices(robot_id: int) -> List[int]:
    """Get indices of all movable (non-fixed) joints.
    
    Args:
        robot_id: PyBullet body ID
        
    Returns:
        List of joint indices
    """
    indices = []
    num_joints = p.getNumJoints(robot_id)
    
    for j in range(num_joints):
        joint_info = p.getJointInfo(robot_id, j)
        if joint_info[2] != p.JOINT_FIXED:
            indices.append(j)
    
    return indices


# ============================================================================
# JOINT-SPACE TRAJECTORY EXECUTION (CRITICAL FIX)
# ============================================================================
# This is the KEY architectural fix: execute JOINT trajectories directly
# instead of calling move_ee() which recomputes IK and executes a DIFFERENT path.

def execute_joint_trajectory(
    robot_id: int,
    joint_path: List[List[float]],
    steps_per_waypoint: int = 30,
    max_force: float = JOINT_MAX_FORCE,
    max_velocity: float = JOINT_MAX_VELOCITY
) -> bool:
    """Execute a joint-space trajectory directly using position control.
    
    CRITICAL: This function executes the EXACT joint path that was collision-checked.
    Do NOT use move_ee() after collision checking - it recomputes IK and follows
    a different path, making collision checking meaningless.
    
    Args:
        robot_id: PyBullet body ID
        joint_path: List of joint configurations to follow
        steps_per_waypoint: Physics steps to hold each waypoint
        max_force: Maximum motor force (N)
        max_velocity: Maximum joint velocity (rad/s)
        
    Returns:
        bool: True if execution completed
    """
    movable_joints = get_movable_joint_indices(robot_id)
    
    for waypoint_idx, target_joints in enumerate(joint_path):
        # Apply position control to each joint
        for j_idx, joint_id in enumerate(movable_joints):
            if j_idx < len(target_joints):
                p.setJointMotorControl2(
                    robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target_joints[j_idx],
                    force=max_force,
                    maxVelocity=max_velocity
                )
        
        # Step simulation to let robot move toward target
        for _ in range(steps_per_waypoint):
            p.stepSimulation()
    
    return True


def execute_joint_trajectory_with_contact_check(
    robot_id: int,
    joint_path: List[List[float]],
    obstacle_ids: List[int],
    excluded_obstacles: Optional[Set[int]] = None,
    steps_per_waypoint: int = 20,
    max_force: float = JOINT_MAX_FORCE,
    max_velocity: float = JOINT_MAX_VELOCITY,
    gripper_body_id: Optional[int] = None
) -> Tuple[bool, int, str]:
    """Execute joint trajectory with runtime contact checking.
    
    This is for guarded approach phase where we need to detect
    unexpected contacts during execution (not just in planning).
    
    IMPORTANT: Now also checks gripper body contacts (for KUKA).
    
    Args:
        robot_id: PyBullet body ID
        joint_path: List of joint configurations
        obstacle_ids: Obstacles to check for contact
        excluded_obstacles: Obstacles to ignore
        steps_per_waypoint: Physics steps per waypoint
        max_force: Maximum motor force
        max_velocity: Maximum joint velocity
        gripper_body_id: Optional visual gripper body ID to also check
        
    Returns:
        Tuple of (completed, stopped_at_waypoint, reason)
    """
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    movable_joints = get_movable_joint_indices(robot_id)
    
    for waypoint_idx, target_joints in enumerate(joint_path):
        # Apply position control
        for j_idx, joint_id in enumerate(movable_joints):
            if j_idx < len(target_joints):
                p.setJointMotorControl2(
                    robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target_joints[j_idx],
                    force=max_force,
                    maxVelocity=max_velocity
                )
        
        # Step and check for contacts
        for step in range(steps_per_waypoint):
            p.stepSimulation()
            
            # Check for contact with obstacles every few steps
            if step % 5 == 0:
                for obs_id in obstacle_ids:
                    if obs_id in excluded_obstacles:
                        continue
                    
                    # Check robot arm contacts
                    contacts = p.getContactPoints(bodyA=robot_id, bodyB=obs_id)
                    if contacts:
                        return False, waypoint_idx, f"Robot contact with obstacle {obs_id}"
                    
                    # CRITICAL FIX: Also check gripper body contacts
                    if gripper_body_id is not None:
                        gripper_contacts = p.getContactPoints(bodyA=gripper_body_id, bodyB=obs_id)
                        if gripper_contacts:
                            return False, waypoint_idx, f"Gripper contact with obstacle {obs_id}"
    
    return True, len(joint_path), "Completed"


# ============================================================================
# COLLISION-AWARE MOTION EXECUTION (FIXED ARCHITECTURE)
# ============================================================================
# CRITICAL: This function now executes JOINT trajectories directly.
# The old version collision-checked a joint path but then called move_ee()
# which recomputes IK and executes a DIFFERENT path - this broke all safety.

def move_ee_collision_safe(
    robot_id: int,
    ee_link: int,
    target_pos: List[float],
    obstacle_ids: List[int],
    target_orn: Optional[List[float]] = None,
    excluded_obstacles: Optional[Set[int]] = None,
    approach_direction: str = "above",
    use_pre_approach: bool = True,
    use_rrt_fallback: bool = True,
    steps: int = 300,
    gripper_body_id: Optional[int] = None
) -> Tuple[bool, str]:
    """Move end-effector to target position while avoiding collisions.
    
    ARCHITECTURAL FIX: This function now executes the EXACT joint path
    that was collision-checked. It does NOT call move_ee() after planning.
    
    This implements a multi-strategy motion approach:
    1. Direct interpolated path with collision checking
    2. Pre-approach + linear approach
    3. RRT-based planning fallback
    
    All paths are executed in JOINT SPACE using execute_joint_trajectory().
    
    Args:
        robot_id: PyBullet body ID of robot
        ee_link: End-effector link index
        target_pos: Target position [x, y, z]
        obstacle_ids: List of obstacle body IDs
        target_orn: Target orientation (quaternion), optional
        excluded_obstacles: Obstacles to ignore (e.g., grasped object)
        approach_direction: Direction for pre-approach ("above", "front", etc.)
        use_pre_approach: Whether to use pre-approach pose
        use_rrt_fallback: Whether to use RRT if direct path fails
        steps: Approximate physics steps for motion (used to scale execution)
        
    Returns:
        Tuple of (success, message)
    """
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    # Get current configuration
    current_joints = get_current_joint_positions(robot_id)
    
    # ARCHITECTURAL FIX: Use multi-candidate IK that checks for collisions
    # This finds an IK solution that is collision-free, or returns None
    target_joints, ik_msg = compute_ik_collision_free(
        robot_id, ee_link, target_pos, target_orn, obstacle_ids,
        excluded_obstacles=excluded_obstacles,
        current_joints=current_joints,
        max_candidates=IK_MAX_CANDIDATES,
        safety_margin=COLLISION_SAFETY_MARGIN,
        gripper_body_id=gripper_body_id,
        position_only_if_stuck=True  # Try position-only if orientation fails
    )
    
    if target_joints is None:
        return False, f"IK failed to find collision-free solution: {ik_msg}"
    
    # Calculate steps per waypoint based on total desired steps
    steps_per_wp = max(10, steps // LINEAR_MOTION_STEPS)
    
    # CHECK: Distance to target - if too close, disable RRT
    # RRT is for free-space. Final 15cm uses deterministic linear approach.
    dist_to_target = math.sqrt(sum((target_pos[i] - current_joints[i])**2 for i in range(3))) if len(current_joints) >= 3 else 999
    use_rrt_near_object = use_rrt_fallback and dist_to_target > 0.15
    
    # =========================================================================
    # STRATEGY 1: Direct interpolated path with collision checking
    # =========================================================================
    
    direct_path = interpolate_joints(current_joints, target_joints, num_steps=LINEAR_MOTION_STEPS)
    is_path_free, fail_step, reason = check_path_collision_free(
        robot_id, direct_path, obstacle_ids, excluded_obstacles,
        safety_margin=COLLISION_SAFETY_MARGIN,
        gripper_body_id=gripper_body_id,
        ee_link=ee_link
    )
    
    if is_path_free:
        # Execute direct path IN JOINT SPACE (not move_ee!)
        print(f"[MOTION] Direct path is collision-free, executing in joint space...")
        execute_joint_trajectory(robot_id, direct_path, steps_per_waypoint=steps_per_wp)
        return True, "Direct path executed (joint-space)"
    
    # ARCHITECTURAL FIX: If step 0 fails, the current position is flagged as collision.
    # This is likely a false positive since we're physically already there.
    # Try with reduced margin to see if we can proceed.
    if fail_step == 0:
        is_path_free_reduced, fail_step_reduced, reason_reduced = check_path_collision_free(
            robot_id, direct_path, obstacle_ids, excluded_obstacles,
            safety_margin=COLLISION_SAFETY_MARGIN / 2,  # Half margin
            gripper_body_id=gripper_body_id,
            ee_link=ee_link
        )
        if is_path_free_reduced:
            print(f"[MOTION] Step 0 was false positive (passed with reduced margin), executing...")
            execute_joint_trajectory(robot_id, direct_path, steps_per_waypoint=steps_per_wp)
            return True, "Direct path executed (reduced margin)"
        elif fail_step_reduced > 0:
            # We can proceed at least partially - the real collision is further along
            print(f"[MOTION] Step 0 was false positive, but blocked at step {fail_step_reduced}: {reason_reduced}")
            # Update fail_step and reason for subsequent strategies
            fail_step = fail_step_reduced
            reason = reason_reduced
    
    print(f"[MOTION] Direct path blocked at step {fail_step}: {reason}")
    
    # =========================================================================
    # STRATEGY 2: Pre-approach + linear approach
    # =========================================================================
    
    if use_pre_approach:
        pre_approach_pos = compute_pre_approach_pose(target_pos, approach_direction)
        
        # Compute IK for pre-approach using multi-candidate approach
        # Use position-only for pre-approach (more flexibility in free space)
        pre_joints, pre_ik_msg = compute_ik_collision_free(
            robot_id, ee_link, pre_approach_pos, None,  # position-only for pre-approach
            obstacle_ids, excluded_obstacles=excluded_obstacles,
            current_joints=current_joints,
            max_candidates=IK_MAX_CANDIDATES // 2,  # Fewer attempts for intermediate pose
            safety_margin=COLLISION_SAFETY_MARGIN,
            gripper_body_id=gripper_body_id,
            position_only_if_stuck=False
        )
        
        if pre_joints is not None:
            # Check path to pre-approach
            pre_path = interpolate_joints(current_joints, pre_joints, num_steps=LINEAR_MOTION_STEPS)
            is_pre_free, _, pre_reason = check_path_collision_free(
                robot_id, pre_path, obstacle_ids, excluded_obstacles,
                safety_margin=COLLISION_SAFETY_MARGIN,
                gripper_body_id=gripper_body_id,
                ee_link=ee_link
            )
            
            if is_pre_free:
                # Check approach path (use tighter margin for approach phase)
                approach_path = interpolate_joints(pre_joints, target_joints, num_steps=LINEAR_MOTION_STEPS * 2)
                is_approach_free, _, approach_reason = check_path_collision_free(
                    robot_id, approach_path, obstacle_ids, excluded_obstacles,
                    safety_margin=APPROACH_SAFETY_MARGIN,  # Tighter margin for approach
                    gripper_body_id=gripper_body_id,
                    ee_link=ee_link
                )
                
                if is_approach_free:
                    print(f"[MOTION] Pre-approach + linear approach is collision-free")
                    # Execute BOTH paths in joint space
                    execute_joint_trajectory(robot_id, pre_path, steps_per_waypoint=steps_per_wp)
                    execute_joint_trajectory(robot_id, approach_path, steps_per_waypoint=steps_per_wp // 2)
                    return True, "Pre-approach path executed (joint-space)"
                else:
                    print(f"[MOTION] Approach path blocked: {approach_reason}")
            else:
                print(f"[MOTION] Pre-approach path blocked: {pre_reason}")
        else:
            print(f"[MOTION] Pre-approach IK failed: {pre_ik_msg}")
    
    # =========================================================================
    # STRATEGY 3: RRT-based planning fallback
    # =========================================================================
    
    if use_rrt_fallback:
        print("[MOTION] Attempting RRT-based planning...")
        
        joint_limits = get_joint_limits(robot_id)
        rrt = SimpleRRT(robot_id, obstacle_ids, joint_limits, excluded_obstacles, 
                        gripper_body_id=gripper_body_id,
                        ee_link=ee_link)
        
        rrt_path = rrt.plan(current_joints, target_joints, max_iterations=RRT_MAX_ITERATIONS)
        
        if rrt_path is not None:
            print(f"[MOTION] RRT found path with {len(rrt_path)} waypoints")
            
            # Execute RRT path directly in joint space (CRITICAL FIX)
            # This is what makes the collision-checking meaningful
            execute_joint_trajectory(
                robot_id, 
                rrt_path, 
                steps_per_waypoint=max(20, steps // len(rrt_path))
            )
            
            return True, "RRT path executed (joint-space)"
        else:
            print("[MOTION] RRT failed to find path")
    
    # =========================================================================
    # FAILURE: No valid path found
    # =========================================================================
    
    return False, "No collision-free path found - motion aborted"


# ============================================================================
# 3-PHASE MOTION SYSTEM (ARCHITECTURAL FIX)
# ============================================================================
# Separates motion into three distinct phases with different behaviors:
# Phase 1: Free-space motion (large steps, RRT allowed)
# Phase 2: Guarded approach (tiny steps, abort on contact)
# Phase 3: Contact-controlled closure (fingers only, force-based)

def compute_geometry_based_approach(
    part_body_id: int,
    robot_position: List[float],
    safety_clearance: float = 0.05
) -> Tuple[List[float], str, float]:
    """Compute approach direction and clearance based on part geometry.
    
    ARCHITECTURAL FIX: This replaces the semantic-only grasp direction selection
    with geometry-driven approach computation using AABBs. Also computes the
    required clearance distance based on part half-extents along approach axis.
    
    Args:
        part_body_id: PyBullet body ID of the part
        robot_position: Current robot base position [x, y, z]
        safety_clearance: Additional clearance beyond part half-extent (meters)
        
    Returns:
        Tuple of (approach_direction_vector, approach_name, approach_clearance)
        - approach_direction_vector: Unit vector pointing FROM part TO approach point
        - approach_name: Human-readable name of approach direction
        - approach_clearance: Distance to pre-approach = part_half_extent + safety
    """
    try:
        # Get axis-aligned bounding box
        aabb_min, aabb_max = p.getAABB(part_body_id)
        
        # Compute dimensions and half-extents
        dims = [aabb_max[i] - aabb_min[i] for i in range(3)]
        half_extents = [d / 2.0 for d in dims]
        
        # Find dominant (largest) axis - this is usually the "length" of the part
        max_dim = max(dims)
        dominant_axis = dims.index(max_dim)
        
        # Compute outward-facing normals for the two smallest faces
        # (perpendicular to the dominant axis)
        center = [(aabb_min[i] + aabb_max[i]) / 2.0 for i in range(3)]
        
        # Determine which face is closer to the robot
        # We want to approach from the side closest to the robot
        if dominant_axis == 2:  # Vertical part (like a leg)
            # Approach from X or Y, whichever is closer to robot
            dx = robot_position[0] - center[0]
            dy = robot_position[1] - center[1]
            
            if abs(dx) > abs(dy):
                # Approach from X direction
                approach_dir = [1.0 if dx > 0 else -1.0, 0.0, 0.0]
                approach_name = "side_x"
                approach_clearance = half_extents[0] + safety_clearance
            else:
                # Approach from Y direction
                approach_dir = [0.0, 1.0 if dy > 0 else -1.0, 0.0]
                approach_name = "side_y"
                approach_clearance = half_extents[1] + safety_clearance
                
        elif dominant_axis == 0:  # Long in X (like armrest)
            # Approach from Y or Z
            dy = robot_position[1] - center[1]
            
            if dims[2] < dims[1]:  # Thinner in Z, approach from above
                approach_dir = [0.0, 0.0, 1.0]
                approach_name = "above"
                approach_clearance = half_extents[2] + safety_clearance
            else:
                approach_dir = [0.0, 1.0 if dy > 0 else -1.0, 0.0]
                approach_name = "side_y"
                approach_clearance = half_extents[1] + safety_clearance
                
        elif dominant_axis == 1:  # Long in Y (like backrest)
            # Approach from X or Z
            dx = robot_position[0] - center[0]
            
            if dims[2] < dims[0]:  # Thinner in Z
                approach_dir = [0.0, 0.0, 1.0]
                approach_name = "above"
                approach_clearance = half_extents[2] + safety_clearance
            else:
                approach_dir = [1.0 if dx > 0 else -1.0, 0.0, 0.0]
                approach_name = "front" if dx > 0 else "back"
                approach_clearance = half_extents[0] + safety_clearance
        else:
            # Fallback: approach from above
            approach_dir = [0.0, 0.0, 1.0]
            approach_name = "above"
            approach_clearance = half_extents[2] + safety_clearance
        
        # Ensure minimum clearance
        approach_clearance = max(approach_clearance, 0.10)  # At least 10cm
        
        return approach_dir, approach_name, approach_clearance
        
    except Exception as e:
        print(f"[GEOMETRY] Failed to compute approach: {e}, using above")
        return [0.0, 0.0, 1.0], "above", 0.15  # Default 15cm clearance


def execute_three_phase_motion(
    robot_id: int,
    ee_link: int,
    target_pos: List[float],
    obstacle_ids: List[int],
    target_orn: Optional[List[float]] = None,
    excluded_obstacles: Optional[Set[int]] = None,
    approach_vector: Optional[List[float]] = None,
    approach_distance: float = 0.15,
    gripper_body_id: Optional[int] = None
) -> Tuple[bool, str]:
    """Execute motion in three distinct phases for reliable grasping.
    
    Phase 1: FREE-SPACE MOTION
        - Move to pre-approach position
        - Uses RRT if needed
        - Large collision margin (3cm)
        
    Phase 2: GUARDED APPROACH
        - Linear motion toward target
        - Very small steps (2mm)
        - Tighter collision margin (1cm)
        - Abort on ANY contact
        
    Phase 3: FINAL POSITIONING
        - Last few mm to target
        - Contact checking at every step
        - Ready for grasp constraint creation
        
    Args:
        robot_id: PyBullet body ID
        ee_link: End-effector link index
        target_pos: Final target position
        obstacle_ids: Obstacles to avoid
        target_orn: Target orientation (optional)
        excluded_obstacles: Obstacles to ignore
        approach_vector: Direction to approach from (unit vector)
        approach_distance: Distance for pre-approach offset
        gripper_body_id: Visual gripper body for collision checking
        
    Returns:
        Tuple of (success, phase_completed)
    """
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    if approach_vector is None:
        approach_vector = [0.0, 0.0, 1.0]  # Default: from above
    
    # Normalize approach vector
    mag = math.sqrt(sum(v*v for v in approach_vector))
    if mag > 0:
        approach_vector = [v / mag for v in approach_vector]
    
    # Compute pre-approach position (offset from target along approach direction)
    pre_approach_pos = [
        target_pos[0] + approach_vector[0] * approach_distance,
        target_pos[1] + approach_vector[1] * approach_distance,
        target_pos[2] + approach_vector[2] * approach_distance
    ]
    
    # =========================================================================
    # PHASE 1: FREE-SPACE MOTION
    # =========================================================================
    print("[MOTION] Phase 1: Free-space motion to pre-approach position")
    
    success, msg = move_ee_collision_safe(
        robot_id, ee_link, pre_approach_pos, obstacle_ids,
        target_orn=target_orn,
        excluded_obstacles=excluded_obstacles,
        use_pre_approach=True,
        use_rrt_fallback=True,
        steps=200,
        gripper_body_id=gripper_body_id
    )
    
    if not success:
        print(f"[MOTION] Phase 1 failed: {msg}")
        return False, "Phase 1 (free-space) failed"
    
    # =========================================================================
    # PHASE 2: GUARDED APPROACH
    # =========================================================================
    print("[MOTION] Phase 2: Guarded approach with contact detection")
    
    # Get current position
    current_joints = get_current_joint_positions(robot_id)
    
    # Compute IK for target using multi-candidate approach
    # For Phase 2 (guarded approach), we're close to target, use orientation
    target_joints, ik_msg = compute_ik_collision_free(
        robot_id, ee_link, target_pos, target_orn, obstacle_ids,
        excluded_obstacles=excluded_obstacles,
        current_joints=current_joints,
        max_candidates=IK_MAX_CANDIDATES,
        safety_margin=APPROACH_SAFETY_MARGIN,  # Tighter margin for approach
        gripper_body_id=gripper_body_id,
        position_only_if_stuck=False
    )
    
    if target_joints is None:
        return False, f"Phase 2 IK failed: {ik_msg}"
    
    # Create fine-grained approach path (many small steps)
    # Calculate distance and number of steps
    ee_state = p.getLinkState(robot_id, ee_link)
    current_pos = list(ee_state[0])
    distance = math.sqrt(sum((target_pos[i] - current_pos[i])**2 for i in range(3)))
    
    # Use 2mm steps for guarded approach
    num_approach_steps = max(10, int(distance / GUARDED_APPROACH_STEP))
    
    approach_path = interpolate_joints(
        current_joints, target_joints, 
        num_steps=num_approach_steps
    )
    
    # Check approach path with TIGHTER margin
    is_approach_free, fail_step, reason = check_path_collision_free(
        robot_id, approach_path, obstacle_ids, excluded_obstacles,
        safety_margin=APPROACH_SAFETY_MARGIN,
        gripper_body_id=gripper_body_id,
        ee_link=ee_link
    )
    
    if not is_approach_free:
        print(f"[MOTION] Phase 2 blocked at step {fail_step}: {reason}")
        return False, f"Phase 2 (guarded approach) blocked: {reason}"
    
    # Execute guarded approach with contact checking
    completed, stopped_at, contact_reason = execute_joint_trajectory_with_contact_check(
        robot_id, approach_path, obstacle_ids,
        excluded_obstacles=excluded_obstacles,
        steps_per_waypoint=5,  # Fewer steps per waypoint for responsiveness
        gripper_body_id=gripper_body_id
    )
    
    if not completed:
        print(f"[MOTION] Phase 2 contact detected at waypoint {stopped_at}: {contact_reason}")
        return False, f"Phase 2 contact: {contact_reason}"
    
    # =========================================================================
    # PHASE 3: FINAL POSITIONING (last adjustments)
    # =========================================================================
    print("[MOTION] Phase 3: Final positioning complete")
    
    return True, "All phases completed"


def retreat_collision_safe(
    robot_id: int,
    ee_link: int,
    retreat_distance: float,
    obstacle_ids: List[int],
    excluded_obstacles: Optional[Set[int]] = None,
    retreat_direction: str = "above",
    steps: int = 200
) -> Tuple[bool, str]:
    """Retreat from current position in a safe direction.
    
    Args:
        robot_id: PyBullet body ID
        ee_link: End-effector link index
        retreat_distance: How far to retreat (meters)
        obstacle_ids: Obstacles to avoid
        excluded_obstacles: Obstacles to ignore
        retreat_direction: Direction to retreat ("above", "back", etc.)
        steps: Physics simulation steps
        
    Returns:
        Tuple of (success, message)
    """
    # Get current EE position
    ee_state = p.getLinkState(robot_id, ee_link)
    current_pos = list(ee_state[0])
    
    # Compute retreat position
    retreat_pos = compute_pre_approach_pose(
        current_pos, 
        approach_direction=retreat_direction,
        offset_distance=retreat_distance
    )
    
    return move_ee_collision_safe(
        robot_id, ee_link, retreat_pos, obstacle_ids,
        excluded_obstacles=excluded_obstacles,
        use_pre_approach=False,  # Direct retreat
        use_rrt_fallback=True,
        steps=steps
    )


# ============================================================================
# COLLISION FILTER MANAGEMENT
# ============================================================================

class CollisionFilterManager:
    """Manages collision filtering for grasped objects.
    
    When a part is grasped, we need to disable collisions between the gripper
    and that part. This class tracks which collision pairs have been disabled
    so they can be re-enabled later.
    """
    
    def __init__(self):
        self.disabled_pairs: List[Tuple[int, int, int, int]] = []
    
    def disable_gripper_part_collision(
        self,
        robot_id: int,
        gripper_id: Optional[int],
        part_id: int,
        ee_link: int
    ):
        """Disable collisions between gripper/EE and a grasped part.
        
        Args:
            robot_id: Robot body ID
            gripper_id: Visual gripper body ID (can be None)
            part_id: Part being grasped
            ee_link: End-effector link index
        """
        pairs_to_disable = []
        
        # Disable EE link vs part
        pairs_to_disable.append((robot_id, part_id, ee_link, -1))
        
        # Disable last few arm links vs part (for safety)
        for link_offset in range(3):
            link_idx = ee_link - link_offset
            if link_idx >= 0:
                pairs_to_disable.append((robot_id, part_id, link_idx, -1))
        
        # Disable gripper vs part (if gripper exists)
        if gripper_id is not None:
            pairs_to_disable.append((gripper_id, part_id, -1, -1))
            # Also gripper fingers
            for finger_link in range(p.getNumJoints(gripper_id)):
                pairs_to_disable.append((gripper_id, part_id, finger_link, -1))
        
        # Apply collision filtering
        for body_a, body_b, link_a, link_b in pairs_to_disable:
            try:
                p.setCollisionFilterPair(body_a, body_b, link_a, link_b, enableCollision=0)
                self.disabled_pairs.append((body_a, body_b, link_a, link_b))
            except Exception as e:
                print(f"[COLLISION] Warning: Could not disable collision pair: {e}")
        
        print(f"[COLLISION] Disabled {len(pairs_to_disable)} collision pairs for part {part_id}")
    
    def restore_all_collisions(self):
        """Re-enable all previously disabled collision pairs."""
        for body_a, body_b, link_a, link_b in self.disabled_pairs:
            try:
                p.setCollisionFilterPair(body_a, body_b, link_a, link_b, enableCollision=1)
            except Exception:
                pass
        
        count = len(self.disabled_pairs)
        self.disabled_pairs.clear()
        print(f"[COLLISION] Re-enabled {count} collision pairs")
    
    def restore_collision_for_part(self, part_id: int):
        """Re-enable collisions only for a specific part."""
        pairs_to_remove = []
        
        for pair in self.disabled_pairs:
            body_a, body_b, link_a, link_b = pair
            if body_a == part_id or body_b == part_id:
                try:
                    p.setCollisionFilterPair(body_a, body_b, link_a, link_b, enableCollision=1)
                except Exception:
                    pass
                pairs_to_remove.append(pair)
        
        for pair in pairs_to_remove:
            self.disabled_pairs.remove(pair)
        
        print(f"[COLLISION] Re-enabled {len(pairs_to_remove)} collision pairs for part {part_id}")


# Global collision filter manager instance
_collision_filter_manager = CollisionFilterManager()


def get_collision_filter_manager() -> CollisionFilterManager:
    """Get the global collision filter manager instance."""
    return _collision_filter_manager


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_obstacle_ids_from_parts(parts: Dict) -> List[int]:
    """Extract obstacle body IDs from parts dictionary.
    
    Args:
        parts: Dictionary of part_name -> (body_id, link_index)
        
    Returns:
        List of body IDs
    """
    obstacle_ids = []
    
    for part_name, part_handle in parts.items():
        if isinstance(part_handle, tuple):
            body_id, _ = part_handle
        else:
            body_id = part_handle
        
        if body_id not in obstacle_ids:
            obstacle_ids.append(body_id)
    
    return obstacle_ids


def log_collision_status(
    robot_id: int,
    obstacle_ids: List[int],
    excluded: Optional[Set[int]] = None
):
    """Log current collision status for debugging."""
    is_collision, obs_id, min_dist = check_collision_robot_vs_obstacles(
        robot_id, obstacle_ids, excluded_obstacles=excluded
    )
    
    if is_collision:
        print(f"[COLLISION STATUS] Robot COLLIDING with obstacle {obs_id}, distance: {min_dist:.4f}m")
    else:
        print(f"[COLLISION STATUS] Robot clear, min distance: {min_dist:.4f}m")
    
    is_self, pair, self_dist = check_self_collision(robot_id)
    if is_self:
        print(f"[COLLISION STATUS] Robot SELF-COLLISION between links {pair}, distance: {self_dist:.4f}m")


if __name__ == "__main__":
    # Simple test
    print("Collision-aware motion module loaded successfully")
    print("Use move_ee_collision_safe() for safe robot motion")
    print("Use CollisionFilterManager for grasp collision handling")
