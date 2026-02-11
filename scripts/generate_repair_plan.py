"""
================================================================================
DESCRIPTION:
    CLI wrapper to test OpenAI repair-plan generation.

USAGE:
    python scripts/generate_repair_plan.py

OUTPUTS:
    Prints plan JSON to stdout.

ARGUMENTS:
    None (uses a fixed test example)
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import json
import logging
import argparse
import os
import sys
from typing import Dict, Any

# Add the parent directory to sys.path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.openai_utils import generate_repair_plan as call_openai_plan, _ensure_remove_before_replace

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def generate_repair_plan(furniture_type: str, damaged_part: str, damage_type: str) -> Dict[str, Any]:
    """
    Wrapper around openai_utils.generate_repair_plan().
    Handles logging and JSON normalization for downstream modules.
    """
    logger.info(f"Requesting OpenAI repair plan for {furniture_type} - {damaged_part} ({damage_type})")


    plan = call_openai_plan(furniture_type, damaged_part, damage_type)

    # Normalize to ensure compatibility with downstream visualization and Webots
    if "repair_sequence" not in plan and "repair_plan" in plan:
        plan = {"repair_sequence": plan["repair_plan"]}

    # --- Post-process: Ensure 'replace' step for broken/missing ---
    if damage_type in ["broken", "missing"]:
        found_replace = False
        for step in plan.get("repair_sequence", []):
            if step.get("action_type", "").lower() == "replace" and step.get("target_part", "") == damaged_part:
                found_replace = True
                break
        if not found_replace:
            # Add a replace step at the end
            max_step = max([s.get("step_id", 0) for s in plan.get("repair_sequence", [])] or [1])
            plan["repair_sequence"].append({
                "step_id": max_step + 1,
                "action_type": "replace",
                "target_part": damaged_part,
                "description": f"Install new {damaged_part}.",
                "tools": ["screwdriver"]
            })
    # Ensure sim-friendly order: remove before replace for each part
    plan["repair_sequence"] = _ensure_remove_before_replace(plan.get("repair_sequence", []))
    return plan


def _cli():
    ap = argparse.ArgumentParser(description="Generate a repair plan via OpenAI API")
    ap.add_argument("--furniture", default="Chair", help="Furniture type")
    ap.add_argument("--part", default="back_frame", help="Damaged part")
    ap.add_argument("--damage", default="broken", help="Damage type")
    args = ap.parse_args()

    plan = generate_repair_plan(args.furniture, args.part, args.damage)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    _cli()