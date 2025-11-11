import pybullet as p
from .sim_connection import step_sim

def load_robot(robot="kuka"):
    if robot == "panda":
        rid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 11
        gripper_joints = [9, 10]
        open_val, close_val = 0.04, 0.0
    else:
        rid = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        ee_link = 6
        gripper_joints = []
        open_val, close_val = None, None
    return rid, ee_link, gripper_joints, open_val, close_val

def open_gripper(robot, joints, val):
    for j in joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)

def close_gripper(robot, joints, val):
    for j in joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)

def move_ee_ik(robot, ee_link, target_pos, target_orn=None, steps=120):
    if target_orn is None:
        target_orn = p.getLinkState(robot, ee_link)[5]
    joint_positions = p.calculateInverseKinematics(robot, ee_link, target_pos, target_orn)
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joint_positions[j], force=200)
    step_sim(steps / 240.0)
