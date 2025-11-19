"""Helper utilities for connecting to and driving a PyBullet simulation.

This module centralizes simple, repeatable actions used by the project's
simulation scripts: establishing a connection, resetting the camera to a
useful viewpoint, stepping the physics engine for a fixed time, and
keeping the GUI open until the user closes it.

Keeping these helpers in one place reduces duplication across demos/tests
and makes example usage clearer.
"""

import pybullet as p
import pybullet_data
import time


def connect(gui: bool = True):
    """Connect to a PyBullet physics server and perform basic setup.

    Args:
        gui: If True start a GUI window (p.GUI). If False use DIRECT mode
             (no on-screen visualization) which is useful for headless runs
             or automated tests.

    Returns:
        The connection id returned by `p.connect` (int).

    Side effects:
        - Adds PyBullet's example data path so URDFs like `plane.urdf`
          can be found.
        - Resets any existing simulation state.
        - Sets Earth's gravity and loads a ground plane.
    """
    cid = p.connect(p.GUI if gui else p.DIRECT)

    # Tell PyBullet where to find example assets (URDFs, meshes, etc.).
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Start from a clean simulation state and enable gravity.
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    # Load a simple ground plane so objects have something to rest on.
    p.loadURDF("plane.urdf")
    return cid


def reset_camera():
    """Set the PyBullet debug visualizer camera to a sensible default.

    The chosen parameters provide an angled, slightly zoomed-out view that
    works well for visualizing a chair on the ground plane used in examples.
    Call this after loading your scene if the default camera is undesirable.
    """
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=40,
        cameraPitch=-35,
        cameraTargetPosition=[0.6, 0.0, 0.4]
    )


def step_sim(seconds: float = 0.4, hz: int = 240):
    """Advance the physics simulation for a short, real-time duration.

    This helper calls `p.stepSimulation()` repeatedly and sleeps between
    steps so the simulation advances at approximately the requested
    frequency. It is primarily intended for demos and simple scripted
    sequences where a few simulation steps are required to reach a stable
    configuration.

    Args:
        seconds: How long to advance the simulation (in real seconds).
        hz: The stepping frequency (Hz). Higher values produce smoother
            physics integration but increase CPU usage.
    """
    for _ in range(int(seconds * hz)):
        p.stepSimulation()
        time.sleep(1.0 / hz)


def keep_window_open():
    """Keep the PyBullet GUI open until the user closes it.

    The function runs a lightweight loop that continues stepping the
    simulation while the connection remains open. It also handles a
    KeyboardInterrupt (Ctrl+C) gracefully by disconnecting.
    """
    print("[INFO] PyBullet simulation complete. Close the window to exit.")
    try:
        # Use `p.isConnected(0)` to check whether the first client is still
        # connected; if the GUI window is closed this will become False.
        while p.isConnected(0):
            p.stepSimulation()
            time.sleep(1.0 / 240)  # 240 Hz loop to avoid busy-waiting
    except KeyboardInterrupt:
        print("[INFO] User interrupted. Closing simulation.")
    finally:
        # Ensure we always disconnect to free resources.
        p.disconnect()
