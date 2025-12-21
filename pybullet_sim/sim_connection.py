"""Helper utilities for connecting to and driving a PyBullet simulation."""

import pybullet as p
import pybullet_data
import time
import numpy as np
from PIL import Image


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


def reset_camera(dist=1.8, yaw=40, pitch=-35, target=[0.6, 0.0, 0.4]):
    """Set the PyBullet debug visualizer camera to a sensible default."""
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target
    )


def save_screenshot(filename, width=640, height=480):
    """Save a screenshot of the current camera view."""
    img = p.getCameraImage(width, height)
    rgba = img[2]  # rgbPixels
    arr = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    rgb = arr[:, :, :3]  # Drop alpha channel
    image = Image.fromarray(rgb)
    image.save(filename)


def step_sim(seconds: float = 0.4, hz: int = 120):
    """Advance the physics simulation.
    
    Args:
        seconds: Duration to simulate
        hz: Physics update frequency (default 120 Hz - optimized for streaming)
    
    NOTE: This function contains time.sleep(). 
    The Flask app (app.py) will OVERRIDE this function dynamically 
    to remove the sleep and capture video frames instead.
    
    Performance Note:
    - Reduced from 240 Hz to 120 Hz for better streaming performance
    - Still 120x real-time simulation, more than sufficient
    - Saves ~50% CPU while maintaining visual quality at 30 FPS
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