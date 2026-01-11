"""
================================================================================
DESCRIPTION:
    OpenAI-backed repair-plan generator. Returns STRICT JSON.
    Falls back to a local deterministic plan when no API key is set
    or on repeated API errors.

USAGE:
    from utils.openai_utils import generate_repair_plan
    plan = generate_repair_plan("Chair", "back", "Fix backrest", "loose")

OUTPUTS:
    Dict containing key "repair_plan" with tools/materials/safety/steps/tips.

ARGUMENTS:
    furniture_type: str
    damaged_part: str
    assembly_step: str
    damage_type: str = "missing"
    Environment:
      OPENAI_API_KEY   required to call OpenAI
      OPENAI_MODEL     optional (default: gpt-4o-mini)
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from scripts.chair_graph import get_dependencies, find_parent

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not found. Using dummy fallback plan.")


def generate_repair_plan(furniture_type: str, damaged_part: str, damage_type: str):
    """
    Generates a repair plan using GPT-4o with STRICT vocabulary for simulation.
    """
    # 1. Fallback if no API key
    if not client:
        return _get_fallback_plan(damaged_part, damage_type)

    # 2. Get Mechanical Context (Dependency Graph)
    deps = get_dependencies(damaged_part)
    parent = find_parent(damaged_part)

    # 3. Construct Context for LLM
    context = {
        "furniture": furniture_type,
        "damaged_part": damaged_part,
        "damage_type": damage_type,
        "dependencies": deps,  # Parts blocking the damaged part
        "parent": parent       # What the damaged part is attached to
    }

    # 4. STRICT System Prompt aligned with sim_plan_executor.py
    system_prompt = (
        "You are an expert furniture repair AI. Generate a structured repair plan JSON.\n"
        "You MUST adhere to these rules for the Robotic Simulator:\n"
        "1. **Action Verbs**: Use ONLY these allowed verbs for 'action_type':\n"
        "   - 'inspect' (Sim: Yellow highlight)\n"
        "   - 'remove'  (Sim: Orange highlight, moves part away)\n"
        "   - 'replace' (Sim: Green highlight, installs new part)\n"
        "   - 'tighten' (Sim: Blue highlight, wiggles tool)\n"
        "   - 'clean'   (Sim: Generic wiggle)\n"
        "2. **Structure**: Return a JSON object with a single key 'repair_sequence'.\n"
        "   'repair_sequence' must be a list of steps. Each step has:\n"
        "      - 'step_id': int\n"
        "      - 'action_type': str (one of the allowed verbs)\n"
        "      - 'target_part': str (exact part name, e.g., 'front_left_leg')\n"
        "      - 'description': str (human readable instructions)\n"
        "      - 'tools': list[str]\n"
        "3. **Logic**: \n"
        "   - If 'dependencies' are listed, you MUST 'remove' them BEFORE working on the 'damaged_part'.\n"
        "   - You MUST 'replace' or 'attach' the dependencies back AFTER fixing the damaged part.\n"
        "   - If damage_type is 'missing' or 'broken', use 'replace' for the damaged part.\n"
        "   - If damage_type is 'loose', use 'tighten'.\n"
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context)},
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        raw_output = response.choices[0].message.content.strip()
        
        # Clean markdown formatting if present (common GPT issue)
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").replace("json", "").strip()

        # Parse JSON
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}")
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON detected in LLM output.")

        json_str = raw_output[json_start:json_end + 1]
        data = json.loads(json_str)

        # Normalize keys just in case
        if "repair_plan" in data and "repair_sequence" not in data:
            data["repair_sequence"] = data["repair_plan"]
            
        return data

    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return _get_fallback_plan(damaged_part, damage_type)


def _get_fallback_plan(part, damage):
    """Deterministic fallback for testing or offline mode."""
    logger.info("Generating fallback deterministic plan.")
    steps = []
    
    # Simple logic: Inspect -> Remove -> Replace/Fix
    steps.append({
        "step_id": 1,
        "action_type": "inspect",
        "target_part": part,
        "description": f"Inspect {part} for {damage} damage.",
        "tools": ["visual_inspection"]
    })

    if damage in ["missing", "broken"]:
        steps.append({
            "step_id": 2,
            "action_type": "remove",
            "target_part": part,
            "description": f"Remove the damaged {part}.",
            "tools": ["screwdriver"]
        })
        steps.append({
            "step_id": 3,
            "action_type": "replace",
            "target_part": part,
            "description": f"Install new {part}.",
            "tools": ["screwdriver"]
        })
    elif damage == "loose":
        steps.append({
            "step_id": 2,
            "action_type": "tighten",
            "target_part": part,
            "description": f"Tighten screws on {part}.",
            "tools": ["screwdriver", "wrench"]
        })
    else:
        # Default generic cleaning step
        steps.append({
            "step_id": 2,
            "action_type": "clean",
            "target_part": part,
            "description": f"Clean and repair {part}.",
            "tools": ["cloth", "filler"]
        })

    return {"repair_sequence": steps}