import matplotlib.pyplot as plt
import os
import json

def render_step_visual(model_path, highlighted_part_idx, save_path, damage_report_path=None):
    """Render visual guidance for repair step with correct orientation and damage labels"""

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

    damaged_parts_with_type = {}
    if damage_report_path and os.path.exists(damage_report_path):
        with open(damage_report_path, 'r') as f:
            damage_report = json.load(f)
        if "detected_pairs" in damage_report:
            for dp in damage_report["detected_pairs"]:
                damaged_parts_with_type[dp["part"]] = dp["damage_type"]

    part_names = list(chair_parts.keys())

    for i, (part_name, part_info) in enumerate(chair_parts.items()):
        x1, y1, x2, y2 = part_info["coords"]
        if i == highlighted_part_idx:
            color = 'red'  # repair focus
            width = 4
        elif part_name in damaged_parts_with_type:
            damage_type = damaged_parts_with_type[part_name]
            color = 'darkred' if damage_type == 'broken' else 'orange'
            width = 3
        else:
            color = part_info["color"]
            width = 1

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             facecolor=color, edgecolor='black', linewidth=width)
        ax.add_patch(rect)

        ax.text((x1 + x2) / 2, (y1 + y2) / 2, part_name.replace('_', ' '),
                ha='center', va='center', fontsize=8, weight='bold')

    if 0 <= highlighted_part_idx < len(part_names):
        part_name = part_names[highlighted_part_idx]
        ax.set_title(f"Repair Focus: {part_name.replace('_', ' ').title()}",
                     fontsize=14, weight='bold')

        part_coords = chair_parts[part_name]["coords"]
        cx = (part_coords[0] + part_coords[2]) / 2
        cy = (part_coords[1] + part_coords[3]) / 2

        ax.annotate('REPAIR THIS PART',
                    xy=(cx, cy),
                    xytext=(cx + 2, cy - 2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red', weight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Chair Width')
    ax.set_ylabel('Chair Height')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visual guidance saved to: {save_path}")
