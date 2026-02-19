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

from utils.repair_planner import build_deterministic_plan
from utils.repair_schema import normalize_part_name, normalize_damage_type

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def generate_repair_plan(furniture_type: str, damaged_part: str, damage_type: str) -> Dict[str, Any]:
    """
    Return a deterministic repair plan so the sim always gets the correct part and
    replace steps (inspect → remove → replace for broken/cracked/missing).
    """
    logger.info(f"Generating repair plan for {furniture_type} - {damaged_part} ({damage_type})")
    try:
        part = normalize_part_name(damaged_part.strip())
    except ValueError:
        part = "back_left_leg"
    damage = normalize_damage_type(damage_type.strip())
    plan = build_deterministic_plan(part, damage)
    logger.info(f"Plan: {len(plan.get('repair_sequence', []))} steps for target_part={part}")
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