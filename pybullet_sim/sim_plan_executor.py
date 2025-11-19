"""Execute high-level repair-plan actions in the PyBullet scene.

This module translates plan steps (strings) into simple robot movements and
visual feedback. It intentionally keeps behavior straightforward so the
mapping between plan intent and simulation behavior is easy to follow.
"""

import pybullet as p
from sim_robot import move_ee, open_gripper, close_gripper
from sim_connection import step_sim
import json


def load_json(path):
    """Load a JSON file from disk and return its parsed content.

    The repair plans used by the demo are simple JSON objects with a
    `repair_sequence` key containing steps. This helper centralizes file
    loading and could be extended to validate plan format.
    """
    with open(path) as f:
        return json.load(f)


# ================================
# Utilities for body/link handling
# ================================


def get_pos(part_handle):
    """Return (pos, orn) for a part.

    The code supports two representations for parts used across the demo:
      - an integer body id (base object), or
      - a tuple (body, link) representing a specific link index inside
        a multi-part URDF.

    The function returns the world-space position and orientation suitable
    for motion planning / visualization.
    """
    if isinstance(part_handle, tuple):
        body, link = part_handle
        ls = p.getLinkState(body, link)
        return ls[0], ls[1]  # pos, orn
    else:
        return p.getBasePositionAndOrientation(part_handle)


def change_color(part_handle, color):
    """Set a visual color for a body or a specific link.

    Uses `-1` for the base link when a single-body id is provided.
    """
    if isinstance(part_handle, tuple):
        body, link = part_handle
        p.changeVisualShape(body, link, rgbaColor=color)
    else:
        p.changeVisualShape(part_handle, -1, rgbaColor=color)


# ===========================
# LLM ACTION INTERPRETER
# ===========================


def classify_action(action: str):
    """Map a natural-language action string to a small set of verbs.

    This lightweight classifier looks for keywords to determine whether the
    plan step is a `remove`, `attach`, `tighten`, `inspect`, `clean`,
    `align`, or `generic` action. Keeping it simple makes behavior
    predictable during demos.
    """
    a = action.lower()

    if any(k in a for k in ["remove","detach","pull","take off","disassemble"]):
        return "remove"
    if any(k in a for k in ["attach","replace","install","put back","assemble"]):
        return "attach"
    if "tighten" in a or "screw" in a:
        return "tighten"
    if "inspect" in a or "check" in a:
        return "inspect"
    if any(k in a for k in ["clean","wipe","sand"]):
        return "clean"
    if any(k in a for k in ["align","position"]):
        return "align"

    return "generic"


# ===========================
# VISUAL FEEDBACK
# ===========================


def highlight_part(part_handle, is_working=False):
    """Quick helper to set a highlight color for a part.

    When `is_working` the part gets a bright yellow color; otherwise a
    muted gray.
    """
    if is_working:
        change_color(part_handle, (1, 1, 0, 1))  # Yellow
    else:
        change_color(part_handle, (0.7, 0.7, 0.7, 1))  # Gray


def recolor(part_handle, color):
    """Alias for `change_color` kept for readability in executor flow."""
    change_color(part_handle, color)


def show_working_animation(robot, ee_link, parts, part, duration=0.5):
    """Play a simple up-down motion while highlighting `part`.

    This provides a compact, visible cue that the robot is performing a
    small local operation (tightening, inspecting, cleaning, etc.).
    """
    pos, _ = get_pos(parts[part])

    # Highlight part to indicate work in progress
    recolor(parts[part], (1, 1, 0, 1))

    # Perform a short up/down sequence to simulate work being done.
    for i in range(10):
        work_pos = [pos[0], pos[1], pos[2] + 0.05 + (i % 2) * 0.02]
        move_ee(robot, ee_link, work_pos)

    # Restore base color after operation
    recolor(parts[part], (0.7, 0.7, 0.7, 1))


# ===========================
# MOVEMENT ROUTINES
# ===========================


def move_to_part(robot, ee_link, parts, part):
    """Move the robot end-effector to a hover pose above `part`.

    A small constant vertical offset is applied to avoid collisions with
    the part when moving into place.
    """
    pos, _ = get_pos(parts[part])
    hover = [pos[0], pos[1], pos[2] + 0.20]
    move_ee(robot, ee_link, hover)


def pick_part(robot, ee_link, parts, part, gripper, close_val):
    """Simulate picking a part by moving down and closing the gripper.

    If no gripper is present (e.g. KUKA model used here), the function
    still moves the end-effector to the pick pose and steps the sim.
    """
    pos, _ = get_pos(parts[part])
    move_ee(robot, ee_link, [pos[0], pos[1], pos[2] + 0.05])
    if gripper:
        close_gripper(robot, gripper, close_val)
    step_sim()


def place_part(robot, ee_link, parts, part, gripper, open_val):
    """Simulate placing a part by moving down and opening the gripper."""
    pos, _ = get_pos(parts[part])
    move_ee(robot, ee_link, [pos[0], pos[1], pos[2] + 0.05])
    if gripper:
        open_gripper(robot, gripper, open_val)
    step_sim()


# ===========================
# MAIN EXECUTION
# ===========================


def execute_step(robot, ee_link, gripper, open_val, close_val, parts, step):
    """Execute a single plan step in the simulation.

    Args:
        robot, ee_link: robot id and end-effector link index
        gripper: list of gripper joint indices (may be empty)
        open_val/close_val: floating positions for open/closed gripper
        parts: mapping of part names to simulation handles
        step: dictionary describing the action (expects 'action',
              'target_part', and 'step_id')
    """
    action = classify_action(step["action"])
    part = step["target_part"]

    print(f"[EXECUTE] Step {step['step_id']}: {step['action']} -> {part} ({action})")

    # Visually mark active part
    if part in parts:
        recolor(parts[part], (1, 1, 0, 1))

    # Move to hover above the target part before performing the action.
    move_to_part(robot, ee_link, parts, part)

    if action == "remove":
        print(f"  -> Removing {part}...")
        pick_part(robot, ee_link, parts, part, gripper, close_val)
        recolor(parts[part], (0.9, 0.6, 0.3, 1))  # Orange indicates removed
        step_sim(0.3)

    elif action == "attach":
        print(f"  -> Attaching {part}...")
        place_part(robot, ee_link, parts, part, gripper, open_val)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))  # Green indicates attached
        step_sim(0.3)

    elif action == "tighten":
        print(f"  -> Tightening {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    elif action == "inspect":
        print(f"  -> Inspecting {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.7, 0.7, 0.7, 1))
        step_sim(0.4)

    elif action == "clean":
        print(f"  -> Cleaning {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    elif action == "align":
        print(f"  -> Aligning {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    else:
        print(f"  -> Processing {part}...")
        show_working_animation(robot, ee_link, parts, part)
        recolor(parts[part], (0.3, 0.8, 0.4, 1))
        step_sim(0.3)

    # Short pause after non-removal steps so viewers can see the result.
    if part in parts and action != "remove":
        step_sim(0.5)
