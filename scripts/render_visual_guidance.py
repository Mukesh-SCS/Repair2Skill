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

def render_step_visual(model_path, highlighted_part_idx, save_path,
                       damage_report_path=None, flip_horizontal=True, flip_vertical=True):
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

    # use only part↔damage pairs as targets
    targets = set()
    if damage_report_path and os.path.exists(damage_report_path):
        with open(damage_report_path, "r") as f:
            rep = json.load(f)
        targets = {dp["part"] for dp in rep.get("detected_pairs", [])}

    for i, (name, (x1,y1,x2,y2)) in enumerate(chair_parts.items()):
        is_focus  = i == highlighted_part_idx
        is_target = name in targets
        color = "red" if (is_focus or is_target) else "lightgray"
        width = 4 if is_focus else (3 if is_target else 1)
        ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                   facecolor=color, edgecolor="black", linewidth=width))
        ax.text((x1+x2)/2, (y1+y2)/2, name.replace("_"," "),
                ha="center", va="center", fontsize=8, weight="bold")

    if 0 <= highlighted_part_idx < len(PARTS):
        name = PARTS[highlighted_part_idx]
        x1,y1,x2,y2 = chair_parts[name]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        ax.set_title(f"Repair Focus: {name.replace('_',' ').title()}",
                     fontsize=14, weight="bold")
        dx = -2 if flip_horizontal else 2
        dy =  2 if flip_vertical   else -2
        ax.annotate("REPAIR THIS PART", xy=(cx,cy), xytext=(cx+dx, cy+dy),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    fontsize=12, color="red", weight="bold")

    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.set_xlabel("Chair Width"); ax.set_ylabel("Chair Height")

    # fixes: real-chair orientation
    if flip_horizontal: ax.invert_xaxis()  # left/right correct
    if flip_vertical:   ax.invert_yaxis()  # back on top, legs at bottom

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Visual guidance saved to: {save_path}")
