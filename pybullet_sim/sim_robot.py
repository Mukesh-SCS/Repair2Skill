import pybullet as p
from sim_connection import step_sim

def load_robot(robot="kuka"):
    if robot == "panda":
        rid = p.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
        ee_link = 11
        gripper = [9,10]
        open_val, close_val = 0.04, 0.0
    else:
        rid = p.loadURDF("kuka_iiwa/model.urdf", [0,0,0], useFixedBase=True)
        ee_link = 6
        gripper = []
        open_val = close_val = None
    return rid, ee_link, gripper, open_val, close_val

def move_ee(robot, ee_link, pos, orn=None, steps=160):
    if orn is None:
        orn = p.getLinkState(robot, ee_link)[5]

    joints = p.calculateInverseKinematics(robot, ee_link, pos, orn)
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joints[j], force=180)
    step_sim(steps / 240)

def open_gripper(robot, joints, val):
    for j in joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)
    step_sim()

def close_gripper(robot, joints, val):
    for j in joints:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=50)
    step_sim()
