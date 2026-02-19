"""
Single source of truth for chair repair pipeline.
Must match: sim_scene.CHAIR_PARTS, sim_plan_executor actions.
"""

# ---------------------------------------------------------------------------
# Step 0 — Chair schema (matches sim_scene.CHAIR_PARTS keys)
# ---------------------------------------------------------------------------
CHAIR_PARTS_LIST = [
    "seat",
    "back",
    "front_left_leg",
    "front_right_leg",
    "back_left_leg",
    "back_right_leg",
    "armrest_left",
    "armrest_right",
]

# Parts that cannot be removed/replaced (structural base)
NON_REPAIRABLE_PARTS = {"seat"}

# Allowed action_type values (matches sim_plan_executor)
ALLOWED_ACTIONS = {"inspect", "remove", "replace", "tighten", "clean"}

# Damage types (detector + normalized)
DAMAGE_TYPES = {"broken", "cracked", "missing", "loose", "dirty", "scratched", "none"}

# Map detector/LLM phrases to canonical part name (Step 4)
PART_NAME_MAPPING = {
    "seat": "seat",
    "back": "back",
    "backrest": "back",
    "front left leg": "front_left_leg",
    "front_left_leg": "front_left_leg",
    "front right leg": "front_right_leg",
    "front_right_leg": "front_right_leg",
    "back left leg": "back_left_leg",
    "rear left leg": "back_left_leg",
    "back_left_leg": "back_left_leg",
    "back right leg": "back_right_leg",
    "rear right leg": "back_right_leg",
    "back_right_leg": "back_right_leg",
    "left arm": "armrest_left",
    "armrest left": "armrest_left",
    "armrest_left": "armrest_left",
    "right arm": "armrest_right",
    "armrest right": "armrest_right",
    "armrest_right": "armrest_right",
}


def normalize_part_name(part: str) -> str:
    """Map any part phrase to canonical name. Returns canonical or raises ValueError."""
    if not part or not isinstance(part, str):
        raise ValueError("Part name must be a non-empty string")
    raw = part.strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in CHAIR_PARTS_LIST:
        return key
    if key in PART_NAME_MAPPING:
        return PART_NAME_MAPPING[key]
    spaced = raw.lower()
    for phrase, canonical in PART_NAME_MAPPING.items():
        if phrase.replace(" ", "_") == key or phrase == spaced:
            return canonical
    raise ValueError(f"Unknown part: {part!r}. Must be one of {CHAIR_PARTS_LIST}")


def normalize_damage_type(damage: str) -> str:
    """Map damage to allowed type. 'scratched' -> 'dirty' for plan logic if needed."""
    if not damage or not isinstance(damage, str):
        return "none"
    d = damage.strip().lower()
    if d in DAMAGE_TYPES:
        return d
    if d in ("dirty", "scratched"):
        return "dirty"
    return "none"
