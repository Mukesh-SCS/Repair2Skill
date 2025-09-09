"""
================================================================================
DESCRIPTION:
    Render a simple chair schematic and highlight ONLY paired target parts.

USAGE:
    from scripts.render_visual_guidance import render_step_visual
    render_step_visual(None, idx, "data/visual_guides/seat_repair_guide.png", "outputs/stage1_parts.json")

OUTPUTS:
    PNG saved to path.

ARGUMENTS:
    model_path: unused
    highlighted_part_idx: index in fixed PARTS list, or -1
    save_path: output PNG path
    damage_report_path: path to Stage-I JSON
Author Info: Mukesh Mani Tripathi
================================================================================
"""
import os, json
import matplotlib.pyplot as plt

PARTS = [
    "seat","back","front_left_leg","front_right_leg",
    "back_left_leg","back_right_leg","armrest_left","armrest_right"
]

def render_step_visual(model_path, highlighted_part_idx, save_path, damage_report_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    chair_parts = {
        "seat": [3,3,7,4],
        "back": [4,1,6,3],
        "front_left_leg":  [3,4,3.5,7],
        "front_right_leg": [6.5,4,7,7],
        "back_left_leg":   [4,4,4.5,7],
        "back_right_leg":  [5.5,4,6,7],
        "armrest_left": [2.5,2,4,2.5],
        "armrest_right":[6,2,7.5,2.5],
    }

    paired_parts = set()
    if damage_report_path and os.path.exists(damage_report_path):
        with open(damage_report_path, "r") as f:
            rep = json.load(f)
        # highlight only the paired targets
        for dp in rep.get("detected_pairs", []):
            paired_parts.add(dp["part"])

    for i, (name, coords) in enumerate(chair_parts.items()):
        x1,y1,x2,y2 = coords
        is_focus = i == highlighted_part_idx
        is_target = name in paired_parts
        color = "red" if (is_focus or is_target) else "lightgray"
        width = 4 if is_focus else (3 if is_target else 1)
        rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, facecolor=color, edgecolor="black", linewidth=width)
        ax.add_patch(rect)
        ax.text((x1+x2)/2, (y1+y2)/2, name.replace("_"," "), ha="center", va="center", fontsize=8, weight="bold")

    if 0 <= highlighted_part_idx < len(PARTS):
        name = PARTS[highlighted_part_idx]
        px1,py1,px2,py2 = chair_parts[name]
        cx, cy = (px1+px2)/2, (py1+py2)/2
        ax.set_title(f"Repair Focus: {name.replace('_',' ').title()}", fontsize=14, weight="bold")
        ax.annotate("REPAIR THIS PART", xy=(cx,cy), xytext=(cx+2,cy-2),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    fontsize=12, color="red", weight="bold")

    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.set_xlabel("Chair Width"); ax.set_ylabel("Chair Height")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Visual guidance saved to: {save_path}")
