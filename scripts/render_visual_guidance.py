import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import json

def render_step_visual(model_path, highlighted_part_idx, save_path, damage_report_path=None):
    """Render visual guidance for repair step with damage detection"""
    
    # Create a simple visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define chair parts with corrected coordinates
    chair_parts = {
        "seat": {"coords": [3, 3, 7, 4], "color": "lightblue"},
        "back": {"coords": [4, 1, 6, 3], "color": "lightgreen"},
        "front_left_leg": {"coords": [6.5, 4, 7, 7], "color": "orange"},  # Swapped for correct orientation
        "front_right_leg": {"coords": [3, 4, 3.5, 7], "color": "orange"},
        "back_left_leg": {"coords": [5.5, 4, 6, 7], "color": "orange"},
        "back_right_leg": {"coords": [4, 4, 4.5, 7], "color": "orange"},
        "armrest_left": {"coords": [6, 2, 7.5, 2.5], "color": "yellow"},  # Swapped
        "armrest_right": {"coords": [2.5, 2, 4, 2.5], "color": "yellow"}
    }
    
    # Draw each part
    damaged_parts = []
    if damage_report_path and os.path.exists(damage_report_path):
        with open(damage_report_path, 'r') as f:
            damage_report = json.load(f)
            if "detected_damages" in damage_report and "detected_parts" in damage_report:
                for i, part in enumerate(damage_report["detected_parts"]):
                    for damage in damage_report["detected_damages"]:
                        if i < len(damage_report["detected_parts"]):
                            damaged_parts.append(part["part"])
    
    for i, (part_name, part_info) in enumerate(chair_parts.items()):
        x1, y1, x2, y2 = part_info["coords"]
        color = "red" if part_name in damaged_parts or i == highlighted_part_idx else part_info["color"]
        width = 3 if part_name in damaged_parts or i == highlighted_part_idx else 1
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           facecolor=color, edgecolor='black', linewidth=width)
        ax.add_patch(rect)
        
        # Add part label
        ax.text((x1+x2)/2, (y1+y2)/2, part_name.replace('_', ' '), 
               ha='center', va='center', fontsize=8, weight='bold')
    
    # Highlight the part that needs repair
    if highlighted_part_idx < len(chair_parts):
        part_name = list(chair_parts.keys())[highlighted_part_idx]
        ax.set_title(f"Repair Focus: {part_name.replace('_', ' ').title()}", 
                    fontsize=14, weight='bold')
        
        # Add arrow pointing to highlighted part
        part_coords = list(chair_parts.values())[highlighted_part_idx]["coords"]
        center_x = (part_coords[0] + part_coords[2]) / 2
        center_y = (part_coords[1] + part_coords[3]) / 2
        
        ax.annotate('REPAIR THIS PART', 
                   xy=(center_x, center_y), 
                   xytext=(center_x + 2, center_y - 2),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, color='red', weight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Chair Width')
    ax.set_ylabel('Chair Height')
    
    # Save the visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visual guidance saved to: {save_path}")