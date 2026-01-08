# ==================== scripts/render_visual_guidance.py ====================
"""
================================================================================
DESCRIPTION:
    Hierarchy-aware visual guide with step annotations.
    Integrates part dependencies (chair_graph) and repair step IDs from
    the LLM-generated plan.

    Highlights:
        - Red: main damaged part
        - Orange: dependent parts
        - Blue/Green: normal parts
        - Displays step numbers from repair plan if available

ARGUMENTS:
    model_path             : Unused placeholder for model-based render.
    highlighted_part_idx   : Index of the main damaged part.
    save_path              : Output PNG path.
    damage_report_path     : Path to detection JSON (optional).
    plan_json_path         : Path to repair plan JSON (optional).

RETURN:
    No return. Writes PNG to save_path.

Author: Mukesh Mani Tripathi
================================================================================
"""

import matplotlib.pyplot as plt
import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.chair_graph import get_dependencies


def render_step_visual(model_path, highlighted_part_idx, save_path,
                       damage_report_path=None, plan_json_path=None,
                       mirror_horizontal: bool = False):
    """Render upright repair guide highlighting damaged and dependent parts."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Chair parts layout matching the 8 detectable parts:
    # seat, back, front_left_leg, front_right_leg, 
    # back_left_leg, back_right_leg, armrest_left, armrest_right
    # NOTE: "cushion" removed as it's not a detectable class
    chair_parts = {
        "seat": {"coords": [3.0, 5.0, 7.0, 6.0], "color": "lightblue"},
        "back": {"coords": [3.0, 6.6, 7.0, 7.8], "color": "lightgreen"},
    
        # Legs positioned vertically below the seat
        "front_left_leg": {"coords": [3.0, 2.0, 3.5, 5.0], "color": "peru"},
        "front_right_leg": {"coords": [6.5, 2.0, 7.0, 5.0], "color": "peru"},
        "back_left_leg": {"coords": [3.6, 2.0, 4.1, 5.0], "color": "sienna"},
        "back_right_leg": {"coords": [5.9, 2.0, 6.4, 5.0], "color": "sienna"},
    
        # Armrests on the sides
        "armrest_left": {"coords": [2.5, 5.5, 3.0, 7.0], "color": "khaki"},
        "armrest_right": {"coords": [7.0, 5.5, 7.5, 7.0], "color": "khaki"},
    }



    if mirror_horizontal:
        mirrored = {}
        for name, info in chair_parts.items():
            x1, y1, x2, y2 = info["coords"]
            mx1 = 10 - x2
            mx2 = 10 - x1
            mirrored[name] = {"coords": [mx1, y1, mx2, y2], "color": info["color"]}
        chair_parts = mirrored

    # --- Load damage report ---
    damaged_parts_with_type = {}
    main_damaged_part = None
    best_score = -1.0

    if damage_report_path and os.path.exists(damage_report_path):
        with open(damage_report_path, "r") as f:
            damage_report = json.load(f)

        pairs = damage_report.get("detected_pairs", [])
        
        # Use smart scoring if available, otherwise fall back to old formula
        scored_pairs = []
        for dp in pairs:
            part = dp.get("part")
            dtype = dp.get("damage_type")
            if not part or not dtype:
                continue

            damaged_parts_with_type[part] = dtype
            
            if 'smart_score' in dp:
                # Use pre-calculated smart score from detection
                score = dp['smart_score']
            else:
                # Fallback for old detection outputs
                overlap = dp.get("overlap_iou", 0.0)
                overlap_bonus = max(0.5, overlap) if overlap > 0.15 else 0.3
                part_conf = float(dp.get("part_confidence", 0.0))
                damage_conf = float(dp.get("damage_confidence", 0.0))
                score = part_conf * damage_conf * overlap_bonus
            
            scored_pairs.append((score, part, dtype))
        
        # Select best scored pair
        if scored_pairs:
            scored_pairs.sort(key=lambda x: -x[0])
            best_score, main_damaged_part, _ = scored_pairs[0]


    # --- Get dependent parts ---
    dependent_parts = []
    if main_damaged_part:
        dependent_parts = get_dependencies(main_damaged_part)

    # --- Load repair plan (for step annotation) ---
    step_map = {}
    if plan_json_path and os.path.exists(plan_json_path):
        with open(plan_json_path, "r") as f:
            plan = json.load(f)
        for step in plan.get("repair_sequence", []):
            part = step.get("target_part", "")
            step_id = step.get("step_id", "")
            if part and step_id:
                if part not in step_map or step_id < step_map[part]:
                    step_map[part] = step_id

    # --- Draw chair parts ---
    for part_name, part_info in chair_parts.items():
        x1, y1, x2, y2 = part_info["coords"]
        if part_name == main_damaged_part:
           color, width = "red", 4
        elif part_name in dependent_parts:
           color, width = "orange", 3
        else:
           color, width = part_info["color"], 1

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             facecolor=color, edgecolor="black", linewidth=width)
        ax.add_patch(rect)

        label = part_name.replace("_", " ")
        if part_name in step_map:
            label += f"\nStep {step_map[part_name]}"
        ax.text((x1 + x2) / 2, (y1 + y2) / 2,
                label, ha="center", va="center",
                fontsize=8, weight="bold", color="black")

    # --- Title + annotation ---
    if main_damaged_part:
        coords = chair_parts[main_damaged_part]["coords"]
        cx = (coords[0] + coords[2]) / 2
        cy = (coords[1] + coords[3]) / 2
        ax.set_title(f"Repair Focus: {main_damaged_part.replace('_', ' ').title()}",
                     fontsize=14, weight="bold", color="red")
        ax.annotate("REPAIR AREA", xy=(cx, cy), xytext=(cx - 1.5, cy + 1.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    fontsize=12, color="red", weight="bold")

    # --- Layout ---
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Chair Width")
    ax.set_ylabel("Chair Height")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Correct upright visual guide saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render visual repair guidance")
    parser.add_argument("--highlighted_part_idx", type=int, default=None, help="Index of the main damaged part (optional, usually inferred)")
    parser.add_argument("--save_path", type=str, required=True, help="Output PNG path")
    parser.add_argument("--damage_report_path", type=str, default=None, help="Path to detection JSON")
    parser.add_argument("--plan_json_path", type=str, default=None, help="Path to repair plan JSON")
    args = parser.parse_args()

    render_step_visual(
        model_path=None,
        highlighted_part_idx=args.highlighted_part_idx,
        save_path=args.save_path,
        damage_report_path=args.damage_report_path,
        plan_json_path=args.plan_json_path,
        mirror_horizontal=False
    )