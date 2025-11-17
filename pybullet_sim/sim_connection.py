import pybullet as p
import pybullet_data
import time

def connect(gui=True):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    return cid

def reset_camera():
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=40,
        cameraPitch=-35,
        cameraTargetPosition=[0.6, 0.0, 0.4]
    )

def step_sim(seconds=0.4, hz=240):
    for _ in range(int(seconds * hz)):
        p.stepSimulation()
        time.sleep(1.0 / hz)
