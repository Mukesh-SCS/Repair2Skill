"""
================================================================================
DESCRIPTION:
    Build a deterministic assembly/repair graph from detected_pairs only.

USAGE:
    from utils.assembly_plan_utils import parse_manual
    graph = parse_manual(None, "outputs/stage1_parts.json")

OUTPUTS:
    {nodes, edges, repair_sequence}
Author Info: Mukesh Mani Tripathi
================================================================================
"""
import json

PRIORITY = [
    "seat","back","front_left_leg","front_right_leg",
    "back_left_leg","back_right_leg","armrest_left","armrest_right"
]

def parse_manual(manual_path, parts_json_path):
    with open(parts_json_path, "r") as f:
        data = json.load(f)

    pairs = [p for p in data.get("detected_pairs", []) if p.get("part") in PRIORITY]
    pairs.sort(key=lambda x: PRIORITY.index(x["part"]))

    graph = {"nodes": [], "edges": [], "repair_sequence": []}
    for i, p in enumerate(pairs):
        action = f"repair_{p['damage_type']}"
        graph["nodes"].append({
            "id": i, "type": "repair_action", "action": action,
            "target_part": p["part"], "confidence": float(p.get("damage_confidence", 0.0))
        })
        graph["repair_sequence"].append({
            "step_id": i, "action": action, "target_part": p["part"],
            "estimated_time": "10-20 minutes", "difficulty": "medium"
        })
        if i > 0:
            graph["edges"].append({"from": i-1, "to": i, "type": "next"})
    return graph
