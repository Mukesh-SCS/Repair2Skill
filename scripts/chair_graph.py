# ==================== scripts/chair_graph.py ====================
"""
Hierarchical part-dependency graph for chair assembly and repair.
Defines which parts depend on others to ensure correct repair order.

IMPORTANT: This graph must match the parts detected by the model:
- PARTS = ["seat", "back", "front_left_leg", "front_right_leg", 
           "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"]
"""

CHAIR_GRAPH = {
    "root": ["seat"],
    
    # Seat is the central piece - everything connects to it
    "seat": ["front_left_leg", "front_right_leg", "back_left_leg", "back_right_leg", 
             "back", "armrest_left", "armrest_right"],
    
    # Legs have no children
    "front_left_leg": [],
    "front_right_leg": [],
    "back_left_leg": [],
    "back_right_leg": [],
    
    # Backrest has no children (simplified from original with frames)
    "back": [],
    
    # Armrests have no children
    "armrest_left": [],
    "armrest_right": [],
    
    # Legacy aliases for backward compatibility
    "front_frame": ["front_left_leg", "front_right_leg"],
    "back_frame": ["back_left_leg", "back_right_leg", "back"],
    "cushion": [],
}


def get_dependencies(part: str):
    """Return all child parts that depend on this part.
    
    For repair: if you need to fix 'part', you may need to remove
    these dependent parts first to access it.
    """
    return CHAIR_GRAPH.get(part, [])


def find_parent(part: str):
    """Find the parent node for a given part.
    
    For repair: the parent is what this part is attached to.
    """
    for parent, children in CHAIR_GRAPH.items():
        if part in children:
            return parent
    return None


def get_all_parts():
    """Return list of all actual chair parts (excluding meta nodes like 'root')."""
    return [
        "seat", "back", 
        "front_left_leg", "front_right_leg",
        "back_left_leg", "back_right_leg",
        "armrest_left", "armrest_right"
    ]