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
          so the executor can operate the gripper. For the KUKA demo there
          is no gripper configured and the returned gripper list is empty.
    """
    if robot == "panda":
        rid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 11
        gripper = [9, 10]
        open_val, close_val = 0.04, 0.0
    else:
        rid = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 6
        gripper = []
        open_val = close_val = None
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
    for j in joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)
    step_sim()


def close_gripper(robot, joints, val):
    """Set gripper joints to `val` to close the gripper and step the sim."""
    for j in joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)
    step_sim()
