"""
================================================================================
DESCRIPTION:
    Rule-based Stage II sequencer that converts detection JSON into a repair graph.

USAGE:
    from utils.assembly_plan_utils import parse_manual
    graph = parse_manual(None, "outputs/stage1_parts.json")

OUTPUTS:
    Dict with nodes, edges, and repair_sequence.

ARGUMENTS:
    manual_path: unused placeholder
    parts_json_path: path to stage1 JSON
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import json


def parse_manual(manual_path, parts_json_path):
    with open(parts_json_path, "r") as f:
        data = json.load(f)

    graph = {"nodes": [], "edges": [], "repair_sequence": []}

    PRIORITY = [
        "seat", "back", "front_left_leg", "front_right_leg",
        "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
    ]

    pairs = data.get("detected_pairs", [])
    pairs = [p for p in pairs if p.get("part") in PRIORITY]
    pairs.sort(key=lambda x: PRIORITY.index(x["part"]))

    node_id = 0
    for p in pairs:
        action = f"repair_{p['damage_type']}"
        graph["nodes"].append({
            "id": node_id,
            "type": "repair_action",
            "action": action,
            "target_part": p["part"],
            "confidence": float(p.get("damage_confidence", 0.0))
        })
        graph["repair_sequence"].append({
            "step_id": node_id,
            "action": action,
            "target_part": p["part"],
            "estimated_time": "10-20 minutes",
            "difficulty": "medium"
        })
        if node_id > 0:
            graph["edges"].append({"from": node_id - 1, "to": node_id, "type": "next"})
        node_id += 1

    return graph
