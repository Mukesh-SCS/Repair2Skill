# ==================== utils/assembly_plan_utils.py ====================
import json
import cv2
import numpy as np
from PIL import Image

def parse_manual(manual_path, parts_json_path):
    """Parse furniture manual and generate assembly graph"""
    
    # Load detected parts
    with open(parts_json_path, 'r') as f:
        parts_data = json.load(f)
    
    # Create assembly graph structure
    assembly_graph = {
        "nodes": [],
        "edges": [],
        "repair_sequence": []
    }
    
    # Process detected damages
    if "detected_damages" in parts_data:
        for i, damage in enumerate(parts_data["detected_damages"]):
            node = {
                "id": i,
                "type": "repair_action",
                "action": f"repair_{damage['type']}",
                "confidence": damage["confidence"]
            }
            assembly_graph["nodes"].append(node)
    
    # Process detected parts
    if "detected_parts" in parts_data:
        for i, part in enumerate(parts_data["detected_parts"]):
            node = {
                "id": len(assembly_graph["nodes"]) + i,
                "type": "part",
                "part_name": part["part"],
                "confidence": part["confidence"]
            }
            assembly_graph["nodes"].append(node)
    
    # Generate repair sequence
    repair_sequence = []
    node_id = 0
    
    # Priority order for chair repair
    priority_parts = [
        "seat", "back", "front_left_leg", "front_right_leg",
        "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
    ]
    
    for part in priority_parts:
        # Check if this part needs repair
        for damage in parts_data.get("detected_damages", []):
            repair_step = {
                "step_id": node_id,
                "action": f"repair_{damage['type']}",
                "target_part": part,
                "estimated_time": "15 minutes",
                "difficulty": "medium"
            }
            repair_sequence.append(repair_step)
            node_id += 1
    
    assembly_graph["repair_sequence"] = repair_sequence
    
    return assembly_graph