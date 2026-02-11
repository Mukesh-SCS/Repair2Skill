"""
sim_connection.py — PyBullet connection, camera, and physics tuning.
Units: meters, Z-up.
"""

import pybullet as p
import logging

logger = logging.getLogger(__name__)

# Physics defaults
DEFAULT_TIME_STEP = 1.0 / 240.0
DEFAULT_SOLVER_ITERATIONS = 150
DEFAULT_LATERAL_FRICTION = 1.0
DEFAULT_SPINNING_FRICTION = 0.1
DEFAULT_ROLLING_FRICTION = 0.1

_client_id = None


def connect(gui=True):
    """Connect to PyBullet. Returns client ID."""
    global _client_id
    if gui:
        _client_id = p.connect(p.GUI)
    else:
        _client_id = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DEFAULT_TIME_STEP, physicsClientId=_client_id)
    p.setPhysicsEngineParameter(
        numSolverIterations=DEFAULT_SOLVER_ITERATIONS,
        physicsClientId=_client_id,
    )
    if hasattr(p, "setDefaultContactBreakingThreshold"):
        p.setDefaultContactBreakingThreshold(0.001, physicsClientId=_client_id)
    logger.info("PyBullet connected (gui=%s)", gui)
    return _client_id


def get_client():
    """Return current physics client ID."""
    return _client_id


def reset_camera(dist=2.4, yaw=55.0, pitch=-25.0, target=(0.35, 0.0, 0.35)):
    """
    Set camera view so both robot (left) and chair (right) are visible.
    target: (x, y, z) look-at point in meters (between robot ~0,0,0 and chair ~0.6,0,0.4).
    """
    cid = get_client()
    if cid is None:
        return
    # PyBullet camera: distance, yaw (deg), pitch (deg), target xyz
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
        physicsClientId=cid,
    )


def step_sim(seconds, hz=240.0, blocking=True):
    """
    Step simulation for given real time.
    seconds: wall-clock duration to simulate.
    hz: simulation step rate (should match physics timeStep).
    blocking: if True, run steps; if False, perform one step (for external loops).
    """
    cid = get_client()
    if cid is None:
        return
    dt = 1.0 / hz
    n_steps = max(1, int(seconds * hz))
    if blocking:
        for _ in range(n_steps):
            p.stepSimulation(physicsClientId=cid)
    else:
        p.stepSimulation(physicsClientId=cid)


def set_physics_defaults(time_step=None, solver_iterations=None):
    """Tune physics parameters."""
    cid = get_client()
    if cid is None:
        return
    if time_step is not None:
        p.setTimeStep(time_step, physicsClientId=cid)
    if solver_iterations is not None:
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iterations,
            physicsClientId=cid,
        )


def set_default_friction(lateral=None, spinning=None, rolling=None):
    """Set default friction for new bodies (affects future createMultiBody)."""
    # PyBullet applies these when creating bodies; we use them in sim_scene
    global _default_lateral, _default_spinning, _default_rolling
    if lateral is not None:
        _default_lateral = lateral
    if spinning is not None:
        _default_spinning = spinning
    if rolling is not None:
        _default_rolling = rolling


# Module-level defaults for friction (used by sim_scene)
_default_lateral = DEFAULT_LATERAL_FRICTION
_default_spinning = DEFAULT_SPINNING_FRICTION
_default_rolling = DEFAULT_ROLLING_FRICTION


def get_default_friction():
    return _default_lateral, _default_spinning, _default_rolling
