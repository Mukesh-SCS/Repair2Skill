# ==================== scripts/chair_graph.py ====================
"""
Hierarchical part-dependency graph for chair assembly and repair.
Defines which parts depend on others to ensure correct repair order.
"""

CHAIR_GRAPH = {
    "root": ["seat"],

    "seat": ["front_frame", "back_frame", "cushion"],
    "front_frame": ["front_left_leg", "front_right_leg"],
    "back_frame": ["back_left_leg", "back_right_leg", "backrest"],

    "cushion": [],
    "front_left_leg": [],
    "front_right_leg": [],
    "back_left_leg": [],
    "back_right_leg": [],
    "backrest": []
}


def get_dependencies(part: str):
    """Return all child parts that depend on this part."""
    return CHAIR_GRAPH.get(part, [])


def find_parent(part: str):
    """Find the parent node for a given part."""
    for parent, children in CHAIR_GRAPH.items():
        if part in children:
            return parent
    return None