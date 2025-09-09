"""
================================================================================
DESCRIPTION:
    Renders a simple chair schematic and highlights damaged or focus parts.

USAGE:
    from scripts.render_visual_guidance import render_step_visual
    render_step_visual(None, 1, "data/visual_guides/back.png", "outputs/stage1_parts.json")

OUTPUTS:
    PNG image saved to the provided path.

ARGUMENTS:
    model_path: unused placeholder
    highlighted_part_idx: int index into part list
    save_path: output image path
    damage_report_path: JSON from detection (stage1)
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os
import json
import matplotlib.pyplot as plt


def render_step_visual(model_path, highlighted_part_idx, save_path, damage_report_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    chair_parts = {
        "seat": {"coords": [3, 3, 7, 4], "color": "lightblue"},
        "back": {"coords": [4, 1, 6, 3], "color": "lightgreen"},
        "front_left_leg": {"coords": [3, 4, 3.5, 7], "color": "orange"},
        "front_right_leg": {"coords": [6.5, 4, 7, 7], "color": "orange"},
        "back_left_leg": {"coords": [4, 4, 4.5, 7], "color": "orange"},
        "back_right_leg": {"coords": [5.5, 4, 6, 7], "color": "orange"},
        "armrest_left": {"coords": [2.5, 2, 4, 2.5], "color": "yellow"},
        "armrest_right": {"coords": [6, 2, 7.5, 2.5], "color": "yellow"},
    }

    damaged_parts = set()
    damage_types = {}

    if damage_report_path and os.path.exists(damage_report_path):
        with open(damage_report_path, "r") as f:
            rep = json.load(f)
        if "detected_pairs" in rep:
            for dp in rep["detected_pairs"]:
                damaged_parts.add(dp["part"])
                damage_types[dp["part"]] = dp.get("damage_type", "unknown")
        elif "detected_parts" in rep and "detected_damages" in rep:
            for part in rep["detected_parts"]:
                damaged_parts.add(part["part"])

    part_names = list(chair_parts.keys())

    for i, (name, info) in enumerate(chair_parts.items()):
        x1, y1, x2, y2 = info["coords"]
        is_focus = i == highlighted_part_idx
        is_dmg = name in damaged_parts
        color = "red" if (is_focus or is_dmg) else info["color"]
        width = 4 if is_focus else (3 if is_dmg else 1)

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=color, edgecolor="black", linewidth=width)
        ax.add_patch(rect)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, name.replace("_", " "), ha="center", va="center", fontsize=8, weight="bold")

    if 0 <= highlighted_part_idx < len(part_names):
        name = part_names[highlighted_part_idx]
        ax.set_title(f"Repair Focus: {name.replace('_', ' ').title()}", fontsize=14, weight="bold")
        px1, py1, px2, py2 = chair_parts[name]["coords"]
        cx = (px1 + px2) / 2
        cy = (py1 + py2) / 2
        ax.annotate("REPAIR THIS PART", xy=(cx, cy), xytext=(cx + 2, cy - 2),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    fontsize=12, color="red", weight="bold")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Chair Width")
    ax.set_ylabel("Chair Height")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visual guidance saved to: {save_path}")
