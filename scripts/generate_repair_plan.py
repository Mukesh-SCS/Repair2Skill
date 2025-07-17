# scripts/generate_repair_plan.py

def generate_repair_plan(furniture_type, damaged_part, damage_type):
    steps = []
    if damage_type == "broken":
        steps = [
            {"step_id": 1, "action": f"Remove the broken {damaged_part}.", "target_part": damaged_part, "estimated_time": "5 mins", "difficulty": "Medium"},
            {"step_id": 2, "action": f"Prepare replacement for {damaged_part}.", "target_part": "replacement_part", "estimated_time": "3 mins", "difficulty": "Easy"},
            {"step_id": 3, "action": f"Attach replacement {damaged_part} securely.", "target_part": damaged_part, "estimated_time": "10 mins", "difficulty": "Medium"},
            {"step_id": 4, "action": "Verify stability and alignment.", "target_part": "chair", "estimated_time": "2 mins", "difficulty": "Easy"},
        ]
    elif damage_type == "loose":
        steps = [
            {"step_id": 1, "action": f"Tighten screws on {damaged_part}.", "target_part": damaged_part, "estimated_time": "5 mins", "difficulty": "Easy"},
            {"step_id": 2, "action": "Check stability.", "target_part": damaged_part, "estimated_time": "2 mins", "difficulty": "Easy"},
        ]
    elif damage_type == "missing":
        steps = [
            {"step_id": 1, "action": f"Order replacement for missing {damaged_part}.", "target_part": damaged_part, "estimated_time": "Varies", "difficulty": "Easy"},
            {"step_id": 2, "action": f"Install replacement {damaged_part}.", "target_part": damaged_part, "estimated_time": "10 mins", "difficulty": "Medium"},
        ]
    elif damage_type == "cracked":
        steps = [
            {"step_id": 1, "action": f"Inspect cracked {damaged_part}.", "target_part": damaged_part, "estimated_time": "3 mins", "difficulty": "Easy"},
            {"step_id": 2, "action": f"Apply wood glue or filler to crack in {damaged_part}.", "target_part": damaged_part, "estimated_time": "15 mins", "difficulty": "Medium"},
            {"step_id": 3, "action": "Clamp and let dry.", "target_part": damaged_part, "estimated_time": "60 mins", "difficulty": "Easy"},
        ]
    elif damage_type == "scratched":
        steps = [
            {"step_id": 1, "action": f"Clean the scratched area on {damaged_part}.", "target_part": damaged_part, "estimated_time": "5 mins", "difficulty": "Easy"},
            {"step_id": 2, "action": f"Sand and repaint or polish {damaged_part}.", "target_part": damaged_part, "estimated_time": "20 mins", "difficulty": "Medium"},
        ]
    else:
        steps = [
            {"step_id": 1, "action": f"Inspect {damaged_part} for repair.", "target_part": damaged_part, "estimated_time": "5 mins", "difficulty": "Easy"},
        ]

    repair_plan = {
        "furniture_type": furniture_type,
        "damaged_part": damaged_part,
        "damage_type": damage_type,
        "repair_sequence": steps
    }
    return repair_plan


if __name__ == "__main__":
    import json
    # Example usage
    plan = generate_repair_plan("Chair", "front_left_leg", "broken")
    print(json.dumps(plan, indent=2))
