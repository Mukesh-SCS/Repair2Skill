"""
sim_robot.py — Franka Panda robot loading, EE motion, gripper control, and grasp constraint.
Uses real gripper joints. Units: meters, Z-up.
"""

import os
import logging
import pybullet as p

from .sim_connection import get_client

logger = logging.getLogger(__name__)

# Panda URDF: try pybullet_data then env
PANDA_URDF = "franka_panda/panda.urdf"


def _find_panda_urdf():
    """Return (data_path, urdf_name) for Panda. data_path is for setAdditionalSearchPath."""
    try:
        import pybullet_data
        data_path = pybullet_data.getDataPath()
        path = os.path.join(data_path, PANDA_URDF)
        if os.path.isfile(path):
            return data_path, PANDA_URDF
        path2 = os.path.join(data_path, "panda.urdf")
        if os.path.isfile(path2):
            return data_path, "panda.urdf"
    except Exception:
        pass
    return None, None


def load_robot(robot_type="panda", base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
    """
    Load Franka Panda robot. Returns (robot_id, ee_link_index, gripper_joint_indices).
    base_orientation: quaternion (x,y,z,w).
    """
    cid = get_client()
    if cid is None:
        raise RuntimeError("PyBullet not connected. Call sim_connection.connect() first.")
    if robot_type != "panda":
        raise ValueError("Only 'panda' is supported.")
    data_path, urdf_name = _find_panda_urdf()
    if not urdf_name:
        raise FileNotFoundError(
            "Panda URDF not found. Install pybullet_data: pip install pybullet_data, "
            "or place franka_panda/panda.urdf in PyBullet data path."
        )
    p.setAdditionalSearchPath(data_path, physicsClientId=cid)
    robot_id = p.loadURDF(
        urdf_name,
        base_position,
        base_orientation,
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
        physicsClientId=cid,
    )
    ee_link = _get_ee_link_index(robot_id, cid)
    gripper_joints = _get_gripper_joint_indices(robot_id, cid)
    logger.info("Loaded Panda: robot_id=%s ee_link=%s gripper_joints=%s", robot_id, ee_link, gripper_joints)
    return robot_id, ee_link, gripper_joints


def _get_ee_link_index(robot_id, cid):
    """Panda: end-effector is typically 'panda_hand' or link 11."""
    n = p.getNumJoints(robot_id, physicsClientId=cid)
    for i in range(n):
        info = p.getJointInfo(robot_id, i, physicsClientId=cid)
        name = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
        if "hand" in name.lower() or "link7" in name or name == "panda_hand":
            return i
    return 11


def _get_gripper_joint_indices(robot_id, cid):
    """Return list of finger joint indices (left, right or single pair)."""
    indices = []
    n = p.getNumJoints(robot_id, physicsClientId=cid)
    for i in range(n):
        info = p.getJointInfo(robot_id, i, physicsClientId=cid)
        name = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
        if "finger" in name.lower() or "gripper" in name.lower():
            indices.append(i)
    if not indices:
        for i in range(n):
            info = p.getJointInfo(robot_id, i, physicsClientId=cid)
            jtype = info[2]
            if jtype == p.JOINT_PRISMATIC:
                indices.append(i)
        indices = indices[-2:] if len(indices) >= 2 else indices
    return indices


def move_ee(robot_id, ee_link, pos, orn=None, cid=None):
    """
    Move end-effector to position pos (x,y,z) and optionally orn (quat xyzw).
    Uses inverse kinematics. Returns joint positions (list) or None on failure.
    """
    if cid is None:
        cid = get_client()
    num_joints = p.getNumJoints(robot_id, physicsClientId=cid)
    if orn is None:
        orn = p.getQuaternionFromEuler([0, 0, 0])
    all_joints = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=cid)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            all_joints.append(i)
    ll = [info[8] for info in (p.getJointInfo(robot_id, j, physicsClientId=cid) for j in all_joints)]
    ul = [info[9] for info in (p.getJointInfo(robot_id, j, physicsClientId=cid) for j in all_joints)]
    jpos = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        pos,
        orn,
        lowerLimits=ll,
        upperLimits=ul,
        jointRanges=[u - l for l, u in zip(ll, ul)],
        maxNumIterations=100,
        residualThreshold=1e-6,
        physicsClientId=cid,
    )
    if jpos is None:
        return None
    for idx, j in enumerate(all_joints):
        p.setJointMotorControl2(
            robot_id, j, p.POSITION_CONTROL, targetPosition=jpos[idx], physicsClientId=cid
        )
    return list(jpos)


def open_gripper(robot_id, gripper_joint_indices, open_val=0.04, cid=None):
    """Set gripper joints to open position (finger separation)."""
    if cid is None:
        cid = get_client()
    for i in gripper_joint_indices:
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL, targetPosition=open_val / 2.0, physicsClientId=cid
        )


def close_gripper(robot_id, gripper_joint_indices, close_val=0.0, cid=None):
    """Set gripper joints to closed position."""
    if cid is None:
        cid = get_client()
    for i in gripper_joint_indices:
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL, targetPosition=close_val, physicsClientId=cid
        )


def make_grasp_constraint(robot_id, ee_link, object_id, object_link_index=-1, cid=None):
    """
    Create a fixed constraint between robot end-effector and object (or object link).
    Returns constraint_id or None on failure.
    """
    if cid is None:
        cid = get_client()
    pos_ee, orn_ee = p.getLinkState(robot_id, ee_link, physicsClientId=cid)[:2]
    if object_link_index < 0:
        pos_obj, orn_obj = p.getBasePositionAndOrientation(object_id, physicsClientId=cid)
    else:
        pos_obj, orn_obj = p.getLinkState(object_id, object_link_index, physicsClientId=cid)[:2]
    # Child frame in object space: where the EE currently is relative to object
    inv_pos, inv_orn = p.invertTransform(pos_obj, orn_obj)
    child_pos, child_orn = p.multiplyTransforms(inv_pos, inv_orn, pos_ee, orn_ee)
    constraint_id = p.createConstraint(
        robot_id,
        ee_link,
        object_id,
        object_link_index,
        p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=child_pos,
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=child_orn,
        physicsClientId=cid,
    )
    return constraint_id


def release_grasp_constraint(constraint_id, cid=None):
    """Remove the grasp constraint."""
    if cid is None:
        cid = get_client()
    if constraint_id is not None:
        p.removeConstraint(constraint_id, physicsClientId=cid)
