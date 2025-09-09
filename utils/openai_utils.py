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
import time
import openai
from dotenv import load_dotenv

load_dotenv()


def _fallback(damaged_part: str):
    return {
        "repair_plan": {
            "tools_needed": ["screwdriver", "PVA wood glue", "clamp"],
            "materials_needed": ["replacement screws"],
            "safety_precautions": ["wear eye protection", "unplug power tools"],
            "steps": [
                {"step_number": 1, "description": f"Inspect and clean the {damaged_part}.", "estimated_time": "3m"},
                {"step_number": 2, "description": f"Tighten or replace fasteners on the {damaged_part}.", "estimated_time": "7m"},
                {"step_number": 3, "description": f"Reinforce and align the {damaged_part}.", "estimated_time": "5m"},
                {"step_number": 4, "description": "Verify stability.", "estimated_time": "2m"}
            ],
            "tips": ["test wobble after each step"],
            "difficulty_level": "easy"
        }
    }


def generate_repair_plan(
    furniture_type: str,
    damaged_part: str,
    assembly_step: str,
    damage_type: str = "missing",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1400,
):
    """
    Uses the OpenAI Chat Completions API to produce a strict-JSON repair plan.
    If OPENAI_API_KEY is missing or the API fails three times, returns a fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback(damaged_part)

    client = openai.OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
You are an expert furniture repair technician.

Input:
- furniture_type: {furniture_type}
- damaged_part: {damaged_part}
- damage_type: {damage_type}
- assembly_step: {assembly_step}

Return ONLY JSON with this schema:
{{
  "repair_plan": {{
    "tools_needed": ["..."],
    "materials_needed": ["..."],
    "safety_precautions": ["..."],
    "steps": [{{"step_number": 1, "description": "...", "estimated_time": "..."}}],
    "tips": ["..."],
    "difficulty_level": "easy|medium|hard"
  }}
}}
"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You produce safe, practical repair plans. Return ONLY JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            obj = json.loads(resp.choices[0].message.content)
            if "repair_plan" in obj and isinstance(obj["repair_plan"].get("steps", []), list):
                return obj
            raise ValueError("Invalid schema from model")
        except Exception as e:
            if attempt == 2:
                out = _fallback(damaged_part)
                out["warning"] = f"fallback_used: {e}"
                return out
            time.sleep(1.5 * (attempt + 1))
