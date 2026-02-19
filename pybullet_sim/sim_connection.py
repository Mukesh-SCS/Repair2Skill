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
    
    # =========================================================================
    # PHYSICS STABILITY SETTINGS (ARCHITECTURAL FIX)
    # =========================================================================
    # These settings improve constraint stability and prevent objects from
    # falling through grippers or jittering during constrained motion.
    #
    # Parameters:
    # - fixedTimeStep: Smaller = more stable but slower (1/240 = 4.17ms)
    # - numSolverIterations: More = more stable constraints (150 is high)
    # - enableConeFriction: 1 = cone friction model (more realistic)
    # - contactBreakingThreshold: Small = keeps contacts longer (0.001m = 1mm)
    # - erp: Error reduction parameter (higher = stiffer constraints)
    # - contactERP: Stiffness of contacts
    try:
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0/240.0,      # Fine timestep for accuracy
            numSolverIterations=150,       # High iterations for stable grasps
            enableConeFriction=1,          # Realistic friction cone
            contactBreakingThreshold=0.001,  # 1mm - keep contacts longer
            erp=0.9,                       # High stiffness for constraints
            contactERP=0.9                 # High stiffness for contacts
        )
    except TypeError:
        # Some PyBullet versions don't support all parameters
        # Try with the basic set that's always supported
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0/240.0,
            numSolverIterations=150,
            enableConeFriction=1,
            erp=0.9
        )
    
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


# Global callback for frame capture during simulation steps
# This allows streaming during repair execution, not just idle loop
_frame_callback = None

def set_frame_callback(callback):
    """Set a callback function to be called during simulation steps.
    
    The callback will be called periodically during step_sim() to allow
    frame capture for streaming. The callback should be fast (< 50ms).
    
    Args:
        callback: A callable that takes no arguments, or None to disable
    """
    global _frame_callback
    _frame_callback = callback


def step_sim(seconds: float = 0.4, hz: int = 240, blocking: bool = True):
    """Advance the physics simulation.
    
    Args:
        seconds: Duration to simulate
        hz: Physics update frequency (default 240 Hz - matches fixedTimeStep)
        blocking: If True, uses time.sleep between steps. If False, runs steps
                  as fast as possible (useful for batch operations).
    
    NOTE: Default hz=240 matches fixedTimeStep=1/240 for consistency.
    This ensures the stepping rate is consistent with physics engine settings.
    Previously was 120 Hz which mismatched the physics timestep.
    
    Performance Note:
    - 240 Hz physics provides accurate constraint and contact behavior
    - Sufficient for real-time streaming at 30 FPS (frame_interval=8)
    """
    global _frame_callback
    num_steps = int(seconds * hz)
    
    # Capture frame every N steps (~30 FPS if hz=240 and frame_interval=8)
    frame_interval = max(1, hz // 30)
    
    if blocking:
        for i in range(num_steps):
            p.stepSimulation()
            
            # Call frame callback periodically for streaming
            if _frame_callback and i % frame_interval == 0:
                try:
                    _frame_callback()
                except Exception:
                    pass  # Don't let frame capture errors stop simulation
            
            time.sleep(1.0 / hz)
    else:
        # Non-blocking mode - run steps as fast as possible
        for i in range(num_steps):
            p.stepSimulation()
            
            # Still call frame callback but less frequently
            if _frame_callback and i % (frame_interval * 2) == 0:
                try:
                    _frame_callback()
                except Exception:
                    pass
        # Brief yield to allow other threads to run
        time.sleep(0.001)


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