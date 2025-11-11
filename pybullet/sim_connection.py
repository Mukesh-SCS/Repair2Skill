import pybullet as p, pybullet_data, time

def connect(gui=True):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    return cid

def reset_camera(target=[0.6, 0.0, 0.4], dist=2.0, yaw=45, pitch=-30):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw,
                                 cameraPitch=pitch, cameraTargetPosition=target)

def step_sim(seconds=1.0, hz=240):
    for _ in range(int(seconds * hz)):
        p.stepSimulation()
        time.sleep(1.0 / hz)
