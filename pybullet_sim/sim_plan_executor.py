"""
sim_plan_executor.py — Execute repair plan steps: inspect, remove, replace, tighten, clean.
Uses scene.detach/attach and gripper grasp constraint.
"""

import logging
import pybullet as p

from .sim_connection import get_client, step_sim
from . import sim_robot
from . import sim_scene

logger = logging.getLogger(__name__)

# Motion constants (meters, seconds) - longer duration so robot motion is visible in GUI
APPROACH_HEIGHT_OFFSET = 0.05
MOVE_DURATION = 1.5
DROP_ZONE = (0.6, 0.5, 0.25)
WIGGLE_AMPLITUDE = 0.02
WIGGLE_STEPS = 30
SIM_HZ = 240.0
OPEN_VAL_DEFAULT = 0.04
CLOSE_VAL_DEFAULT = 0.0


def execute_step(robot_id, ee_link, gripper_joints, scene, step, open_val=None, close_val=None, cid=None):
    """
    Execute one repair step. Returns True if step was handled (even if target_part missing).
    """
    if cid is None:
        cid = get_client()
    open_val = open_val if open_val is not None else OPEN_VAL_DEFAULT
    close_val = close_val if close_val is not None else CLOSE_VAL_DEFAULT
    action = (step.get("action_type") or "").strip().lower()
    target_part = (step.get("target_part") or "").strip()
    if not target_part:
        logger.warning("Step %s has no target_part, skipping", step.get("step_id"))
        return True
    if target_part not in sim_scene.CHAIR_PARTS:
        logger.warning("Step %s target_part '%s' not in chair parts, skipping", step.get("step_id"), target_part)
        return True

    if action == "inspect":
        return _do_inspect(robot_id, ee_link, scene, target_part, cid)
    if action == "remove":
        return _do_remove(robot_id, ee_link, gripper_joints, scene, target_part, open_val, close_val, cid)
    if action == "replace":
        return _do_replace(robot_id, ee_link, gripper_joints, scene, target_part, open_val, close_val, cid)
    if action in ("tighten", "clean"):
        return _do_tighten_clean(robot_id, ee_link, scene, target_part, cid)
    logger.warning("Unknown action_type '%s', skipping step", action)
    return True


def _approach_part(robot_id, ee_link, scene, part_name, cid, height_offset=0.0):
    """Move EE above the part (optional height_offset in z)."""
    pos, _ = scene.get_part_pose(part_name)
    if pos is None:
        return False
    approach = [pos[0], pos[1], pos[2] + APPROACH_HEIGHT_OFFSET + height_offset]
    orn = p.getQuaternionFromEuler([0, 0, 0])
    sim_robot.move_ee(robot_id, ee_link, approach, orn, cid)
    step_sim(MOVE_DURATION, SIM_HZ, blocking=True)
    return True


def _do_inspect(robot_id, ee_link, scene, target_part, cid):
    """Move EE near part and highlight it."""
    scene.recolor(target_part, (1.0, 1.0, 0.3, 1.0))
    ok = _approach_part(robot_id, ee_link, scene, target_part, cid)
    step_sim(0.5, SIM_HZ, blocking=True)
    scene.recolor(target_part, (0.6, 0.45, 0.3, 1.0))
    return ok


def _do_remove(robot_id, ee_link, gripper_joints, scene, target_part, open_val, close_val, cid):
    """Move to part, close gripper, detach, then immediately grasp (no step between detach and grasp)."""
    body_id = scene.part_body_id(target_part)
    if body_id is None:
        logger.warning("remove: no body for part %s", target_part)
        return False
    pos, orn = scene.get_part_pose(target_part)
    if pos is None:
        return False
    # 1. Move robot to part while part is still attached
    approach = [pos[0], pos[1], pos[2] + APPROACH_HEIGHT_OFFSET]
    orn_flat = p.getQuaternionFromEuler([0, 0, 0])
    jpos = sim_robot.move_ee(robot_id, ee_link, approach, orn_flat, cid)
    if jpos is None:
        logger.warning("remove: IK failed for approach to %s", target_part)
    step_sim(MOVE_DURATION, SIM_HZ, blocking=True)
    sim_robot.open_gripper(robot_id, gripper_joints, open_val, cid)
    step_sim(0.3, SIM_HZ, blocking=True)
    down = [pos[0], pos[1], pos[2] + 0.03]
    sim_robot.move_ee(robot_id, ee_link, down, orn_flat, cid)
    step_sim(MOVE_DURATION, SIM_HZ, blocking=True)
    sim_robot.close_gripper(robot_id, gripper_joints, close_val, cid)
    step_sim(0.3, SIM_HZ, blocking=True)
    # 2. Detach part from chair (no physics step yet)
    scene.detach(target_part)
    # 3. Immediately create grasp constraint so part is held by robot before any physics step
    grasp_id = sim_robot.make_grasp_constraint(robot_id, ee_link, body_id, -1, cid)
    if grasp_id is None:
        logger.warning("remove: grasp constraint failed for %s", target_part)
        return False
    step_sim(0.5, SIM_HZ, blocking=True)
    # 4. Move to drop zone and release
    drop_above = [DROP_ZONE[0], DROP_ZONE[1], DROP_ZONE[2] + 0.15]
    sim_robot.move_ee(robot_id, ee_link, drop_above, orn_flat, cid)
    step_sim(MOVE_DURATION, SIM_HZ, blocking=True)
    sim_robot.move_ee(robot_id, ee_link, list(DROP_ZONE), orn_flat, cid)
    step_sim(0.4, SIM_HZ, blocking=True)
    sim_robot.release_grasp_constraint(grasp_id, cid)
    sim_robot.open_gripper(robot_id, gripper_joints, open_val, cid)
    step_sim(0.4, SIM_HZ, blocking=True)
    return True


def _do_replace(robot_id, ee_link, gripper_joints, scene, target_part, open_val, close_val, cid):
    """Spawn replacement, grasp it, move to original pose, attach, release."""
    # If the part is still attached (LLM skipped "remove"), detach it first so we don't have two bodies for the same part.
    if scene.constraints.get(target_part):
        logger.info("replace: part %s still attached; detaching first (missing 'remove' step)", target_part)
        scene.detach(target_part)
        step_sim(0.5, SIM_HZ, blocking=True)
    original_pos, original_orn = scene.original_poses.get(target_part, (None, None))
    if original_pos is None:
        original_pos, original_orn = scene.get_part_pose(target_part)
        if original_pos is None:
            # Use CHAIR_PARTS default pose for this part
            if target_part in sim_scene.CHAIR_PARTS:
                center = sim_scene.CHAIR_PARTS[target_part][0]
                original_pos, original_orn = list(center), [0, 0, 0, 1]
            else:
                logger.warning("replace: no original pose for %s", target_part)
                return False
    replacement_id = scene.spawn_replacement(target_part)
    if replacement_id is None:
        logger.warning("replace: spawn_replacement failed for %s", target_part)
        return False
    step_sim(0.2, SIM_HZ, blocking=True)
    bin_above = [PARTS_BIN_POS[0], PARTS_BIN_POS[1], PARTS_BIN_POS[2] + 0.1]
    orn_flat = p.getQuaternionFromEuler([0, 0, 0])
    sim_robot.move_ee(robot_id, ee_link, bin_above, orn_flat, cid)
    step_sim(MOVE_DURATION, SIM_HZ, blocking=True)
    sim_robot.open_gripper(robot_id, gripper_joints, open_val, cid)
    step_sim(0.15, SIM_HZ, blocking=True)
    sim_robot.move_ee(robot_id, ee_link, list(PARTS_BIN_POS), orn_flat, cid)
    step_sim(0.25, SIM_HZ, blocking=True)
    sim_robot.close_gripper(robot_id, gripper_joints, close_val, cid)
    step_sim(0.15, SIM_HZ, blocking=True)
    grasp_id = sim_robot.make_grasp_constraint(robot_id, ee_link, replacement_id, -1, cid)
    if grasp_id is None:
        logger.warning("replace: grasp failed for replacement %s", target_part)
        return False
    step_sim(0.2, SIM_HZ, blocking=True)
    above_original = [original_pos[0], original_pos[1], original_pos[2] + 0.12]
    sim_robot.move_ee(robot_id, ee_link, above_original, orn_flat, cid)
    step_sim(MOVE_DURATION, SIM_HZ, blocking=True)
    at_original = [original_pos[0], original_pos[1], original_pos[2] + 0.02]
    sim_robot.move_ee(robot_id, ee_link, at_original, original_orn, cid)
    step_sim(0.4, SIM_HZ, blocking=True)
    sim_robot.release_grasp_constraint(grasp_id, cid)
    sim_robot.open_gripper(robot_id, gripper_joints, open_val, cid)
    step_sim(0.1, SIM_HZ, blocking=True)
    p.resetBasePositionAndOrientation(replacement_id, original_pos, original_orn, physicsClientId=cid)
    step_sim(0.1, SIM_HZ, blocking=True)
    scene.attach(target_part, (original_pos, original_orn), body_id=replacement_id)
    step_sim(0.3, SIM_HZ, blocking=True)
    return True


def _do_tighten_clean(robot_id, ee_link, scene, target_part, cid):
    """Move to part and wiggle."""
    ok = _approach_part(robot_id, ee_link, scene, target_part, cid)
    if not ok:
        return False
    pos, _ = scene.get_part_pose(target_part)
    if pos is None:
        return False
    for i in range(WIGGLE_STEPS):
        dx = WIGGLE_AMPLITUDE if i % 2 == 0 else -WIGGLE_AMPLITUDE
        sim_robot.move_ee(robot_id, ee_link, [pos[0] + dx, pos[1], pos[2] + APPROACH_HEIGHT_OFFSET], p.getQuaternionFromEuler([0, 0, 0]), cid)
        step_sim(0.05, SIM_HZ, blocking=True)
    return True


# For replace: parts bin position (must match sim_scene)
PARTS_BIN_POS = sim_scene.PARTS_BIN_POS
