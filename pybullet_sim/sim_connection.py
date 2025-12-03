"""Helper utilities for connecting to and driving a PyBullet simulation."""

import pybullet as p
import pybullet_data
import time


def connect(gui: bool = False):
    """Connect to a PyBullet physics server and perform basic setup.

    Args:
        gui: If True start a GUI window. If False use DIRECT mode (headless).
    """
    cid = p.connect(p.GUI if gui else p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    return cid


def reset_camera():
    """Set the PyBullet debug visualizer camera to a sensible default."""
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=40,
        cameraPitch=-35,
        cameraTargetPosition=[0.6, 0.0, 0.4]
    )


def step_sim(seconds: float = 0.4, hz: int = 240):
    """Advance the physics simulation.
    
    NOTE: This function contains time.sleep(). 
    The Flask app (app.py) will OVERRIDE this function dynamically 
    to remove the sleep and capture video frames instead.
    """
    for _ in range(int(seconds * hz)):
        p.stepSimulation()
        time.sleep(1.0 / hz)


def keep_window_open():
    """Keep the PyBullet GUI open until the user closes it."""
    print("[INFO] PyBullet simulation complete. Close the window to exit.")
    try:
        while p.isConnected(0):
            p.stepSimulation()
            time.sleep(1.0 / 240)
    except KeyboardInterrupt:
        print("[INFO] User interrupted. Closing simulation.")
    finally:
        p.disconnect()