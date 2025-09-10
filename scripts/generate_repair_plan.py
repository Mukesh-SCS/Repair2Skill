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
from utils.openai_utils import generate_repair_plan

if __name__ == "__main__":
    plan = generate_repair_plan(
        furniture_type="Chair",
        damaged_part="front_left_leg",
        assembly_step="Attach leg with 4 screws",
        damage_type="loose",
    )
    print(json.dumps(plan, indent=2))
