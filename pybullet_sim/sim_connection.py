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


def save_screenshot(filename, width=640, height=480, dist=1.8, yaw=40, pitch=-35, target=[0.6, 0.0, 0.4]):
    """Save a screenshot of the current camera view.
    
    Works in both GUI and headless (DIRECT) mode.
    In headless mode, uses explicit camera parameters.
    """
    try:
        # Try to get camera image with explicit view matrix (works in DIRECT mode)
        # Compute view matrix from camera parameters
        import math
        
        # Convert angles to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        # Calculate camera position
        cam_x = target[0] + dist * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = target[1] + dist * math.cos(pitch_rad) * math.cos(yaw_rad)
        cam_z = target[2] + dist * math.sin(pitch_rad)
        camera_pos = [cam_x, cam_y, cam_z]
        
        # Up vector (typically [0, 0, 1] for z-up)
        up_vector = [0, 0, 1]
        
        # Get camera image with explicit view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=up_vector
        )
        
        # Projection matrix
        aspect = width / height
        near = 0.01
        far = 100.0
        fov = 60.0
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        # Get camera image
        img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        rgba = img[2]  # rgbPixels
        arr = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
        rgb = arr[:, :, :3]  # Drop alpha channel
        image = Image.fromarray(rgb)
        image.save(filename)
    except Exception as e:
        # Fallback: try simple getCameraImage (might work in some PyBullet versions)
        try:
            img = p.getCameraImage(width, height)
            rgba = img[2]
            arr = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            rgb = arr[:, :, :3]
            image = Image.fromarray(rgb)
            image.save(filename)
        except Exception as e2:
            print(f"[WARN] Failed to save screenshot: {e2}")


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