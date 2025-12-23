"""Robot loading and simple motion helpers for the demo.

This module provides a minimal abstraction for loading a robot model,
moving its end-effector using inverse kinematics, and operating a
gripper if available. The functions are intentionally small and
well-documented to make the simulation flow easy to follow.
"""

import pybullet as p
from sim_connection import step_sim


def load_robot(robot: str = "kuka"):
    """Load a robot URDF and return helper info.

    Args:
        robot: Identifier string. Supported values: "panda" (Franka Panda)
               or any other value (defaults to a KUKA IIWA model).

    Returns:
        A tuple: (robot_id, ee_link_index, gripper_joint_list, open_val, close_val)

    Notes:
        - For the Panda robot we expose gripper joints and open/close values
          so the executor can operate the gripper. For the KUKA, we now 
          provide simulated gripper joints for manipulation.
    """
    if robot == "panda":
        rid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 11
        gripper = [9, 10]
        open_val, close_val = 0.04, 0.0
    else:
        # KUKA IIWA with gripper
        rid = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 6
        
        # For Kuka, we'll simulate a gripper with joints 6 and 7 (if available)
        # These may not actually exist in the URDF, but we can still try to control them
        # Alternatively, we define a virtual gripper mechanism
        num_joints = p.getNumJoints(rid)
        
        # Try to find gripper joints or use virtual ones
        gripper = []
        open_val, close_val = 0.04, 0.0
        
        # Try to get actual gripper joints if they exist
        for joint_idx in range(num_joints):
            try:
                joint_info = p.getJointInfo(rid, joint_idx)
                joint_name = joint_info[1].decode('utf-8').lower()
                if 'finger' in joint_name or 'gripper' in joint_name:
                    gripper.append(joint_idx)
            except:
                pass
        
        # If no gripper joints found, create virtual ones (we'll use joint positions to fake it)
        # For the Kuka model, we can assume joints 6+ might be available
        if not gripper and num_joints > 7:
            gripper = [6, 7]  # Virtual gripper joints
            
    return rid, ee_link, gripper, open_val, close_val


def move_ee(robot, ee_link, pos, orn=None, steps=160):
    """Move the robot end-effector to `pos` (and optional `orn`).

    This convenience wrapper calculates joint targets via inverse
    kinematics and then commands position controllers on all joints.

    Args:
        robot: PyBullet body id for the robot.
        ee_link: index of the end-effector link.
        pos: target position [x,y,z].
        orn: optional target orientation (quaternion). If omitted the
             current EE orientation is preserved.
        steps: how many physics steps to wait after commanding joints.
    """
    if orn is None:
        orn = p.getLinkState(robot, ee_link)[5]

    try:
        joints = p.calculateInverseKinematics(robot, ee_link, pos, orn)
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

    # Only set motors for joints that exist in both the IK result and robot
    max_set = min(num_joints, len(joints))
    for j in range(max_set):
        try:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joints[j], force=180)
        except Exception as e:
            print(f"[WARNING] Failed to set joint {j}: {e}")
    # Convert requested step count into seconds at 240 Hz stepping used
    # elsewhere in the demo.
    step_sim(steps / 240)


def open_gripper(robot, joints, val):
    """Set gripper joints to `val` to open the gripper and step the sim."""
    if not joints or len(joints) == 0:
        # No actual gripper joints, just step the simulation
        step_sim(0.1)
        return
    
    for j in joints:
        try:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)
        except Exception as e:
            print(f"[WARNING] Could not open gripper joint {j}: {e}")
    step_sim()


def close_gripper(robot, joints, val):
    """Set gripper joints to `val` to close the gripper and step the sim."""
    if not joints or len(joints) == 0:
        # No actual gripper joints, just step the simulation
        step_sim(0.1)
        return
    
    for j in joints:
        try:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)
        except Exception as e:
            print(f"[WARNING] Could not close gripper joint {j}: {e}")
    step_sim()
