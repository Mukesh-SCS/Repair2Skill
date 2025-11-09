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
from typing import Dict, Any
from utils.openai_utils import generate_repair_plan as call_openai_plan

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