# ==================== scripts/render_visual_guidance.py ====================
"""
Render a realistic-looking visual repair guide: chair diagram with highlighted
damaged part, dependent parts, and step-by-step plan. Uses wood-like colors,
rounded shapes, and clear typography for presentation.

Supports:
  - damage_report_path: detection JSON (detected_pairs) or damage report (damaged_part, damage_type)
  - plan_json_path: repair_sequence for step numbers and step list
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.colors as mcolors

# Repo root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.chair_graph import get_dependencies

# ---------------------------------------------------------------------------
# Realistic chair layout (side-view style, proportional)
# coords: [x_left, y_bottom, x_right, y_top] in figure units
# Wood-like base colors (hex)
# ---------------------------------------------------------------------------
CHAIR_LAYOUT = {
    "seat": {
        "coords": [2.8, 4.2, 7.2, 4.8],
        "color": "#C4A574",  # wood tan
        "label": "Seat",
    },
    "back": {
        "coords": [2.8, 4.8, 7.2, 7.6],
        "color": "#8B7355",  # darker wood
        "label": "Back",
    },
    "front_left_leg": {
        "coords": [2.9, 1.2, 3.5, 4.2],
        "color": "#6B4423",  # brown
        "label": "Front L Leg",
    },
    "front_right_leg": {
        "coords": [6.5, 1.2, 7.1, 4.2],
        "color": "#6B4423",
        "label": "Front R Leg",
    },
    "back_left_leg": {
        "coords": [3.5, 1.2, 4.1, 4.2],
        "color": "#5D3A1A",  # dark brown
        "label": "Back L Leg",
    },
    "back_right_leg": {
        "coords": [5.9, 1.2, 6.5, 4.2],
        "color": "#5D3A1A",
        "label": "Back R Leg",
    },
    "armrest_left": {
        "coords": [2.2, 4.5, 2.8, 6.2],
        "color": "#9B7E5C",  # medium wood
        "label": "Armrest L",
    },
    "armrest_right": {
        "coords": [7.2, 4.5, 7.8, 6.2],
        "color": "#9B7E5C",
        "label": "Armrest R",
    },
}


def _load_main_damaged_part_and_type(damage_report_path):
    """Load main damaged part and damage type from detection or damage report JSON."""
    if not damage_report_path or not os.path.exists(damage_report_path):
        return None, None
    with open(damage_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # New pipeline: damage report has damaged_part, damage_type
    if "damaged_part" in data and data["damaged_part"]:
        return data["damaged_part"].strip(), (data.get("damage_type") or "damage").strip()
    # Legacy: detection with detected_pairs
    pairs = data.get("detected_pairs", [])
    if not pairs:
        return None, None
    scored = []
    for p in pairs:
        part = (p.get("part") or "").strip()
        dtype = (p.get("damage_type") or "damage").strip()
        if not part:
            continue
        score = float(
            p.get("smart_score")
            or (float(p.get("part_confidence", 0)) * float(p.get("damage_confidence", 0)))
        )
        scored.append((score, part, dtype))
    if not scored:
        return None, None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored[0][2]


def _load_step_map(plan_json_path):
    """Load part -> step_id from repair_sequence."""
    step_map = {}
    if not plan_json_path or not os.path.exists(plan_json_path):
        return step_map
    with open(plan_json_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    for step in plan.get("repair_sequence", []):
        part = (step.get("target_part") or "").strip()
        step_id = step.get("step_id")
        if part and step_id is not None:
            if part not in step_map or step_id < step_map[part]:
                step_map[part] = step_id
    return step_map


def _load_step_list(plan_json_path):
    """Load ordered list of steps (description + target_part) for sidebar."""
    steps = []
    if not plan_json_path or not os.path.exists(plan_json_path):
        return steps
    with open(plan_json_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    for step in plan.get("repair_sequence", []):
        step_id = step.get("step_id")
        desc = (step.get("description") or "").strip() or step.get("action_type", "")
        part = (step.get("target_part") or "").strip()
        action = (step.get("action_type") or "").strip()
        steps.append((step_id, action, part, desc))
    return sorted(steps, key=lambda x: (x[0] if x[0] is not None else 0))


def render_step_visual(
    model_path,
    highlighted_part_idx,
    save_path,
    damage_report_path=None,
    plan_json_path=None,
    mirror_horizontal: bool = False,
):
    """Render a realistic visual repair guide with chair diagram, legend, and steps."""

    chair_parts = dict(CHAIR_LAYOUT)
    if mirror_horizontal:
        mirrored = {}
        for name, info in chair_parts.items():
            x1, y1, x2, y2 = info["coords"]
            mx1, mx2 = 10 - x2, 10 - x1
            mirrored[name] = {**info, "coords": [mx1, y1, mx2, y2]}
        chair_parts = mirrored

    main_damaged_part, damage_type = _load_main_damaged_part_and_type(damage_report_path)
    dependent_parts = get_dependencies(main_damaged_part) if main_damaged_part else []
    step_map = _load_step_map(plan_json_path)
    step_list = _load_step_list(plan_json_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    fig.patch.set_facecolor("#F5F0E8")  # warm off-white
    ax.set_facecolor("#F5F0E8")

    # Draw each part as rounded rectangle with optional highlight
    for part_name, info in chair_parts.items():
        x1, y1, x2, y2 = info["coords"]
        w, h = x2 - x1, y2 - y1
        color = info["color"]
        edge_color = "#3D2B1F"
        linewidth = 1.2

        if part_name == main_damaged_part:
            color = "#C0392B"  # strong red for damage
            edge_color = "#922B21"
            linewidth = 2.5
            # Damage fill with diagonal hatch for realism
            patch = FancyBboxPatch(
                (x1, y1), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=0.92,
                hatch="///",
                fill=True,
            )
        elif part_name in dependent_parts:
            color = "#D35400"  # orange
            edge_color = "#A04000"
            linewidth = 1.8
            patch = FancyBboxPatch(
                (x1, y1), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=0.9,
            )
        else:
            patch = FancyBboxPatch(
                (x1, y1), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=0.95,
            )
        ax.add_patch(patch)

        label = info.get("label", part_name.replace("_", " ").title())
        if part_name in step_map:
            label += f" (Step {step_map[part_name]})"
        text_color = "white" if part_name == main_damaged_part or part_name in dependent_parts else "#2C1810"
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2,
            label,
            ha="center", va="center",
            fontsize=8, weight="bold", color=text_color,
            wrap=True,
        )

    # Title and subtitle
    if main_damaged_part:
        title = f"Repair focus: {main_damaged_part.replace('_', ' ').title()}"
        if damage_type and damage_type.lower() not in ("none", "damage"):
            title += f" — {damage_type.upper()}"
        ax.set_title(title, fontsize=14, weight="bold", color="#2C1810", pad=12)
    else:
        ax.set_title("Chair repair guide", fontsize=14, weight="bold", color="#2C1810", pad=12)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#C0392B", edgecolor="#922B21", label="Damaged (repair focus)", hatch="///"),
        mpatches.Patch(facecolor="#D35400", edgecolor="#A04000", label="Related part"),
        mpatches.Patch(facecolor="#C4A574", edgecolor="#3D2B1F", label="Other parts"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.95)

    # Step-by-step list (right side)
    if step_list:
        step_text = "Repair steps:\n" + "\n".join(
            f"  {s[0]}. {s[1].title()}: {s[2].replace('_', ' ')}" for s in step_list[:8]
        )
        ax.text(
            0.98, 0.5, step_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDC3C7", alpha=0.95),
            family="monospace",
        )

    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(0.8, 8.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[OK] Visual guide saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render visual repair guidance")
    parser.add_argument("--highlighted_part_idx", type=int, default=None)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--damage_report_path", type=str, default=None)
    parser.add_argument("--plan_json_path", type=str, default=None)
    args = parser.parse_args()
    render_step_visual(
        model_path=None,
        highlighted_part_idx=args.highlighted_part_idx,
        save_path=args.save_path,
        damage_report_path=args.damage_report_path,
        plan_json_path=args.plan_json_path,
        mirror_horizontal=False,
    )
