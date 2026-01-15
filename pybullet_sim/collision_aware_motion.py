"""Collision-aware motion planning for PyBullet robot simulation.

This module provides collision checking and safe motion primitives to prevent
the robot arm from colliding with chair parts during repair operations.

COLLISION CHECKING STRATEGY:
----------------------------
1. Use PyBullet's native collision detection (getClosestPoints, getContactPoints)
2. Check robot links vs environment obstacles (chair parts)
3. Check robot self-collision (link vs link)
4. Use safety margin (default 1cm) for conservative collision detection

MOTION PLANNING STRATEGY:
-------------------------
1. Pre-approach: Move to safe offset position before approaching target
2. Linear approach: Small-step interpolation with collision checks
3. Post-action retreat: Reverse path before any lateral motion

If direct path fails, use lightweight sampling-based fallback (RRT-like).

Author: Collision-aware motion module for Repair2Skill
"""

import pybullet as p
import math
import random
from typing import List, Tuple, Optional, Dict, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

# Safety margin for collision detection (meters)
COLLISION_SAFETY_MARGIN = 0.01  # 1cm

# Maximum distance to check for collisions
COLLISION_CHECK_DISTANCE = 0.05  # 5cm

# Number of interpolation steps for linear motion
LINEAR_MOTION_STEPS = 20

# RRT parameters
RRT_MAX_ITERATIONS = 100
RRT_STEP_SIZE = 0.15  # radians
RRT_GOAL_BIAS = 0.3  # 30% chance to sample goal directly

# Pre-approach offset distance (meters)
PRE_APPROACH_OFFSET = 0.15  # 15cm above target

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


def check_collision_robot_vs_obstacle(
    robot_id: int,
    obstacle_id: int,
    robot_links: Optional[List[int]] = None,
    safety_margin: float = COLLISION_SAFETY_MARGIN
) -> Tuple[bool, float]:
    """Check if robot collides with a single obstacle.
    
    Args:
        robot_id: PyBullet body ID of the robot
        obstacle_id: PyBullet body ID of the obstacle
        robot_links: Specific links to check (None = all)
        safety_margin: Minimum allowed distance
        
    Returns:
        Tuple of (is_collision, min_distance)
    """
    if robot_links is None:
        robot_links = get_robot_arm_links(robot_id)
    
    min_distance = float('inf')
    
    for link_idx in robot_links:
        try:
            # Get closest points between robot link and obstacle
            contacts = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=obstacle_id,
                distance=COLLISION_CHECK_DISTANCE,
                linkIndexA=link_idx,
                linkIndexB=-1  # Base link of obstacle
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
    excluded_obstacles: Optional[Set[int]] = None
) -> Tuple[bool, int, float]:
    """Check if robot collides with any obstacle in the list.
    
    Args:
        robot_id: PyBullet body ID of the robot
        obstacle_ids: List of obstacle body IDs
        robot_links: Specific links to check (None = all)
        safety_margin: Minimum allowed distance
        excluded_obstacles: Set of obstacle IDs to skip (e.g., grasped object)
        
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
            
        is_collision, distance = check_collision_robot_vs_obstacle(
            robot_id, obs_id, robot_links, safety_margin
        )
        
        if is_collision:
            return True, obs_id, distance
    
    return False, -1, float('inf')


def check_self_collision(
    robot_id: int,
    safety_margin: float = COLLISION_SAFETY_MARGIN
) -> Tuple[bool, Tuple[int, int], float]:
    """Check for robot self-collision.
    
    Args:
        robot_id: PyBullet body ID of the robot
        safety_margin: Minimum allowed distance
        
    Returns:
        Tuple of (is_collision, (link1, link2), min_distance)
    """
    arm_links = get_robot_arm_links(robot_id)
    min_distance = float('inf')
    collision_pair = (-1, -1)
    
    # Check each pair of non-adjacent links
    for i, link_a in enumerate(arm_links):
        for link_b in arm_links[i+2:]:  # Skip adjacent links
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
    safety_margin: float = COLLISION_SAFETY_MARGIN
) -> Tuple[bool, str]:
    """Check if a joint configuration is collision-free.
    
    This function temporarily sets joint positions, checks collisions,
    and returns whether the configuration is valid.
    
    Args:
        robot_id: PyBullet body ID of the robot
        joint_positions: List of joint position values
        obstacle_ids: List of obstacle body IDs to check against
        excluded_obstacles: Set of obstacles to ignore (e.g., grasped object)
        check_self: Whether to check self-collision
        safety_margin: Minimum allowed distance
        
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
    
    # Check collisions
    try:
        # Check robot vs obstacles
        is_collision, obs_id, _ = check_collision_robot_vs_obstacles(
            robot_id, obstacle_ids, 
            excluded_obstacles=excluded_obstacles,
            safety_margin=safety_margin
        )
        
        if is_collision:
            # Restore original states
            for i, j in enumerate(joint_indices):
                p.resetJointState(robot_id, j, original_states[i])
            return False, f"Collision with obstacle {obs_id}"
        
        # Check self-collision
        if check_self:
            is_self_collision, pair, _ = check_self_collision(robot_id, safety_margin)
            if is_self_collision:
                # Restore original states
                for i, j in enumerate(joint_indices):
                    p.resetJointState(robot_id, j, original_states[i])
                return False, f"Self-collision between links {pair}"
        
    finally:
        # Always restore original states
        for i, j in enumerate(joint_indices):
            p.resetJointState(robot_id, j, original_states[i])
    
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


def check_path_collision_free(
    robot_id: int,
    joint_path: List[List[float]],
    obstacle_ids: List[int],
    excluded_obstacles: Optional[Set[int]] = None,
    safety_margin: float = COLLISION_SAFETY_MARGIN
) -> Tuple[bool, int, str]:
    """Check if entire path is collision-free.
    
    Args:
        robot_id: PyBullet body ID of the robot
        joint_path: List of joint configurations along the path
        obstacle_ids: Obstacles to check against
        excluded_obstacles: Obstacles to ignore
        safety_margin: Minimum allowed distance
        
    Returns:
        Tuple of (is_collision_free, failing_step_index, reason)
    """
    for i, config in enumerate(joint_path):
        is_free, reason = is_configuration_collision_free(
            robot_id, config, obstacle_ids, 
            excluded_obstacles=excluded_obstacles,
            safety_margin=safety_margin
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
    """
    
    def __init__(
        self,
        robot_id: int,
        obstacle_ids: List[int],
        joint_limits: List[Tuple[float, float]],
        excluded_obstacles: Optional[Set[int]] = None
    ):
        """Initialize the RRT planner.
        
        Args:
            robot_id: PyBullet body ID
            obstacle_ids: List of obstacle body IDs
            joint_limits: List of (min, max) tuples for each joint
            excluded_obstacles: Obstacles to ignore during collision checking
        """
        self.robot_id = robot_id
        self.obstacle_ids = obstacle_ids
        self.joint_limits = joint_limits
        self.excluded_obstacles = excluded_obstacles or set()
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
        """Check if configuration is collision-free."""
        is_free, _ = is_configuration_collision_free(
            self.robot_id,
            config,
            self.obstacle_ids,
            excluded_obstacles=self.excluded_obstacles,
            check_self=True
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
        if not self._is_collision_free(start_config):
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


# ============================================================================
# COLLISION-AWARE MOTION EXECUTION
# ============================================================================

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
    steps: int = 300
) -> Tuple[bool, str]:
    """Move end-effector to target position while avoiding collisions.
    
    This implements a 3-stage motion strategy:
    1. Pre-approach: Move to safe offset position
    2. Linear approach: Collision-checked interpolation
    3. If direct path fails, use RRT planner
    
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
        steps: Physics steps per motion segment
        
    Returns:
        Tuple of (success, message)
    """
    from sim_robot import move_ee
    
    if excluded_obstacles is None:
        excluded_obstacles = set()
    
    # Get current configuration
    current_joints = get_current_joint_positions(robot_id)
    
    # Compute IK for target
    if target_orn is not None:
        target_joints = p.calculateInverseKinematics(robot_id, ee_link, target_pos, target_orn)
    else:
        target_joints = p.calculateInverseKinematics(robot_id, ee_link, target_pos)
    
    target_joints = list(target_joints)[:len(current_joints)]
    
    # =========================================================================
    # STRATEGY 1: Direct interpolated path with collision checking
    # =========================================================================
    
    direct_path = interpolate_joints(current_joints, target_joints, num_steps=LINEAR_MOTION_STEPS)
    is_path_free, fail_step, reason = check_path_collision_free(
        robot_id, direct_path, obstacle_ids, excluded_obstacles
    )
    
    if is_path_free:
        # Execute direct path
        print(f"[MOTION] Direct path is collision-free, executing...")
        move_ee(robot_id, ee_link, target_pos, orn=target_orn, steps=steps)
        return True, "Direct path executed"
    
    print(f"[MOTION] Direct path blocked at step {fail_step}: {reason}")
    
    # =========================================================================
    # STRATEGY 2: Pre-approach + linear approach
    # =========================================================================
    
    if use_pre_approach:
        pre_approach_pos = compute_pre_approach_pose(target_pos, approach_direction)
        
        # Check if pre-approach is reachable directly
        if target_orn is not None:
            pre_joints = p.calculateInverseKinematics(robot_id, ee_link, pre_approach_pos, target_orn)
        else:
            pre_joints = p.calculateInverseKinematics(robot_id, ee_link, pre_approach_pos)
        
        pre_joints = list(pre_joints)[:len(current_joints)]
        
        # Check path to pre-approach
        pre_path = interpolate_joints(current_joints, pre_joints, num_steps=LINEAR_MOTION_STEPS)
        is_pre_free, _, pre_reason = check_path_collision_free(
            robot_id, pre_path, obstacle_ids, excluded_obstacles
        )
        
        if is_pre_free:
            # Check approach path
            approach_path = interpolate_joints(pre_joints, target_joints, num_steps=LINEAR_MOTION_STEPS)
            is_approach_free, _, approach_reason = check_path_collision_free(
                robot_id, approach_path, obstacle_ids, excluded_obstacles
            )
            
            if is_approach_free:
                print(f"[MOTION] Pre-approach + linear approach is collision-free")
                # Execute: current -> pre-approach -> target
                move_ee(robot_id, ee_link, pre_approach_pos, orn=target_orn, steps=steps // 2)
                move_ee(robot_id, ee_link, target_pos, orn=target_orn, steps=steps // 2)
                return True, "Pre-approach path executed"
            else:
                print(f"[MOTION] Approach path blocked: {approach_reason}")
        else:
            print(f"[MOTION] Pre-approach path blocked: {pre_reason}")
    
    # =========================================================================
    # STRATEGY 3: RRT-based planning fallback
    # =========================================================================
    
    if use_rrt_fallback:
        print("[MOTION] Attempting RRT-based planning...")
        
        joint_limits = get_joint_limits(robot_id)
        rrt = SimpleRRT(robot_id, obstacle_ids, joint_limits, excluded_obstacles)
        
        path = rrt.plan(current_joints, target_joints)
        
        if path is not None:
            print(f"[MOTION] RRT found path with {len(path)} waypoints")
            
            # Execute path waypoint by waypoint
            for i, waypoint_joints in enumerate(path[1:], 1):  # Skip first (current pos)
                # Get approximate cartesian position for this configuration
                # Temporarily set joints to get EE position
                num_joints = p.getNumJoints(robot_id)
                joint_idx = 0
                for j in range(num_joints):
                    joint_info = p.getJointInfo(robot_id, j)
                    if joint_info[2] != p.JOINT_FIXED:
                        if joint_idx < len(waypoint_joints):
                            p.resetJointState(robot_id, j, waypoint_joints[joint_idx])
                        joint_idx += 1
                
                # Get EE position
                ee_state = p.getLinkState(robot_id, ee_link)
                waypoint_pos = ee_state[0]
                
                # Execute motion to this waypoint
                move_ee(robot_id, ee_link, list(waypoint_pos), orn=target_orn, 
                       steps=max(50, steps // len(path)))
            
            return True, "RRT path executed"
        else:
            print("[MOTION] RRT failed to find path")
    
    # =========================================================================
    # FAILURE: No valid path found
    # =========================================================================
    
    return False, "No collision-free path found - motion aborted"


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
