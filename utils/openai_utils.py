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

# ==================== scripts/openai_utils.py ====================
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


def generate_repair_plan(furniture_type: str, damaged_part: str, damage_type: str) -> dict:
    """
    Generate a repair plan using OpenAI API with chair dependency context.
    No fallback plan is used — requires a valid API key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Build structured context using chair_graph
    dependencies = get_dependencies(damaged_part)
    parent = find_parent(damaged_part)

    context = {
        "furniture_type": furniture_type,
        "damaged_part": damaged_part,
        "damage_type": damage_type,
        "dependencies": dependencies,
        "parent_part": parent,
    }

    system_prompt = (
        "You are a robotics repair planner for furniture. "
        "Use the provided chair dependency context to create a repair sequence. "
        "Each step must include: step_id, action, target_part, tool, estimated_time, difficulty. "
        "Ensure the sequence follows mechanical dependencies: "
        "disassemble dependent parts before repair, reassemble after. "
        "Return **only valid JSON** with key 'repair_sequence'."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context)},
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        raw_output = response.choices[0].message.content.strip()
        logger.info(f"Raw LLM output: {raw_output[:200]}")

        # Clean markdown formatting if present
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").replace("json", "").strip()

        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}")
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON detected in LLM output.")

        json_str = raw_output[json_start:json_end + 1]
        data = json.loads(json_str)

        if "repair_plan" in data:
            data = {"repair_sequence": data["repair_plan"]}
        elif "repair_sequence" not in data:
            raise ValueError("Missing 'repair_sequence' in model output.")

        return data

    except Exception as e:
        logger.error(f"OpenAI repair plan generation failed: {e}")
        raise