"""
Step 1–3: Damage report, deterministic planning, and strict validation.
Plan contains steps only for ONE damaged_part. Seat cannot be remove/replace.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.repair_schema import (
    ALLOWED_ACTIONS,
    CHAIR_PARTS_LIST,
    DAMAGE_TYPES,
    NON_REPAIRABLE_PARTS,
    normalize_damage_type,
    normalize_part_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Damage report (normalize detector output)
# ---------------------------------------------------------------------------
def build_damage_report(detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    From detector output produce strict damage report.
    detection_result must have "detected_pairs" (list of {part, damage_type, ...}).
    Returns: {"damaged_part": str, "damage_type": str, "confidence": float}
    """
    pairs = detection_result.get("detected_pairs") or []
    if not pairs:
        return {"damaged_part": "", "damage_type": "none", "confidence": 0.0}

    # Use best-scored pair
    best = None
    best_score = -1.0
    for p in pairs:
        part = (p.get("part") or "").strip()
        damage = (p.get("damage_type") or "none").strip().lower()
        score = float(p.get("smart_score") or p.get("part_confidence", 0) * p.get("damage_confidence", 0))
        if not part:
            continue
        try:
            canonical_part = normalize_part_name(part)
            if canonical_part in NON_REPAIRABLE_PARTS:
                continue  # skip seat as "damaged" target
            damage_n = normalize_damage_type(damage)
            if score > best_score:
                best_score = score
                best = {"damaged_part": canonical_part, "damage_type": damage_n, "confidence": score}
        except ValueError:
            continue
    if best is None and pairs:
        # Fallback: use first pair and force normalize
        p = pairs[0]
        part = (p.get("part") or "back_left_leg").strip() or "back_left_leg"
        damage = (p.get("damage_type") or "broken").strip().lower()
        try:
            canonical_part = normalize_part_name(part)
            damage_n = normalize_damage_type(damage)
            best = {
                "damaged_part": canonical_part,
                "damage_type": damage_n,
                "confidence": float(p.get("smart_score", 0.5)),
            }
        except ValueError:
            best = {"damaged_part": "back_left_leg", "damage_type": "broken", "confidence": 0.5}
    return best or {"damaged_part": "back_left_leg", "damage_type": "none", "confidence": 0.0}


# ---------------------------------------------------------------------------
# Step 2 — Deterministic plan (no LLM)
# ---------------------------------------------------------------------------
def build_deterministic_plan(damaged_part: str, damage_type: str) -> Dict[str, Any]:
    """
    Convert (damaged_part, damage_type) → minimal plan.
    - broken/cracked/missing → inspect → remove → replace
    - loose → inspect → tighten
    - dirty/scratched → inspect → clean
    - none → inspect only
    """
    part = damaged_part.strip()
    damage = normalize_damage_type(damage_type)
    try:
        part = normalize_part_name(part)
    except ValueError:
        part = "back_left_leg"

    if part in NON_REPAIRABLE_PARTS:
        # Seat: only inspect
        return {
            "repair_sequence": [
                {"step_id": 1, "action_type": "inspect", "target_part": part, "description": f"Inspect {part}.", "tools": []},
            ]
        }

    steps: List[Dict[str, Any]] = []
    step_id = 1
    steps.append({
        "step_id": step_id,
        "action_type": "inspect",
        "target_part": part,
        "description": f"Inspect {part}.",
        "tools": [],
    })
    step_id += 1

    if damage in ("broken", "cracked", "missing"):
        steps.append({
            "step_id": step_id,
            "action_type": "remove",
            "target_part": part,
            "description": f"Remove damaged {part}.",
            "tools": ["screwdriver", "gripper"],
        })
        step_id += 1
        steps.append({
            "step_id": step_id,
            "action_type": "replace",
            "target_part": part,
            "description": f"Replace {part} with a new part.",
            "tools": ["screwdriver", "gripper"],
        })
    elif damage == "loose":
        steps.append({
            "step_id": step_id,
            "action_type": "tighten",
            "target_part": part,
            "description": f"Tighten {part}.",
            "tools": ["screwdriver", "wrench"],
        })
    elif damage in ("dirty", "scratched"):
        steps.append({
            "step_id": step_id,
            "action_type": "clean",
            "target_part": part,
            "description": f"Clean {part}.",
            "tools": ["cloth"],
        })
    # else: none → inspect only

    return {"repair_sequence": steps}


# ---------------------------------------------------------------------------
# Step 3 — Validator (reject bad plans, return deterministic on failure)
# ---------------------------------------------------------------------------
def validate_plan(plan: Dict[str, Any], damaged_part: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate plan. Every step must have target_part == damaged_part.
    Seat cannot appear in remove/replace. action_type must be in allowed list.
    If invalid, return (False, deterministic_plan). Else (True, plan).
    """
    seq = plan.get("repair_sequence") or plan.get("repair_plan") or []
    if not isinstance(seq, list):
        return False, build_deterministic_plan(damaged_part, "broken")

    damage_type = "broken"
    for step in seq:
        if not isinstance(step, dict):
            return False, build_deterministic_plan(damaged_part, damage_type)
        target = (step.get("target_part") or "").strip()
        action = (step.get("action_type") or "").strip().lower()
        if target != damaged_part:
            logger.warning("Validator: step target_part %r != damaged_part %r", target, damaged_part)
            return False, build_deterministic_plan(damaged_part, damage_type)
        if action not in ALLOWED_ACTIONS:
            logger.warning("Validator: invalid action_type %r", action)
            return False, build_deterministic_plan(damaged_part, damage_type)
        if target in NON_REPAIRABLE_PARTS and action in ("remove", "replace"):
            logger.warning("Validator: seat cannot be remove/replace")
            return False, build_deterministic_plan(damaged_part, damage_type)
        if "damage_type" in step:
            damage_type = step.get("damage_type", damage_type)

    return True, plan


def get_valid_plan(damaged_part: str, damage_type: str, llm_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a valid plan. If llm_plan is provided and valid, use it; else use deterministic.
    """
    if llm_plan:
        valid, plan = validate_plan(llm_plan, damaged_part)
        if valid:
            return plan
    return build_deterministic_plan(damaged_part, damage_type)
