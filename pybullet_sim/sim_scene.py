"""Scene creation utilities used by the simulation demos.

The scene functions are intentionally tiny: create simple geometric
blocks or spawn a procedural chair and return a mapping of named
parts to simulation handles (body id, link index) used by the executor.
"""

import pybullet as p

def block(size, pos, color):
    """Create a simple static box and return its body id.

    Args:
        size: [x, y, z] full extents of the box.
        pos: base position to place the box (center of mass).
        color: RGBA tuple for the visual shape.
    """
    half = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    # Mass 0 means static (unmovable) object
    return p.createMultiBody(0, col, vis, pos)


def spawn_simple_chair(damaged_part_name):
    """Procedurally generate a chair using simple boxes.
    
    This replaces the URDF loading mechanism to ensure the chair 
    always looks correct regardless of file paths or mesh issues.
    
    Args:
        damaged_part_name: The name of the part to highlight in RED.
        
    Returns:
        dict: Mapping of part_name -> (body_id, -1)
    """
    parts = {}
    
    # ---------------------------------------------------------
    # 1. Configuration (Dimensions & Colors)
    # ---------------------------------------------------------
    # Base Position of the chair in the world
    bx, by, bz = 0.6, 0.0, 0.0
    
    # Dimensions (Meters)
    seat_w, seat_d, seat_h = 0.45, 0.45, 0.05
    leg_w, leg_h = 0.05, 0.45
    back_h = 0.5
    back_thickness = 0.05
    
    # Colors
    c_wood = [0.6, 0.4, 0.2, 1]   # Brown
    c_dark = [0.5, 0.35, 0.15, 1] # Darker Brown
    c_dmg  = [1, 0, 0, 1]         # Red (Damage)

    def get_color(name):
        return c_dmg if name == damaged_part_name else c_wood

    # ---------------------------------------------------------
    # 2. Build Parts (Calculated relative to Base)
    # ---------------------------------------------------------
    
    # -- SEAT --
    # Placed on top of the legs
    seat_z = bz + leg_h + (seat_h / 2)
    parts["seat"] = block(
        [seat_d, seat_w, seat_h], 
        [bx, by, seat_z], 
        get_color("seat")
    )

    # -- LEGS --
    # Centers of the legs
    dx = (seat_d / 2) - (leg_w / 2)
    dy = (seat_w / 2) - (leg_w / 2)
    leg_z = bz + (leg_h / 2)

    # Front is +X, Back is -X
    parts["front_left_leg"] = block(
        [leg_w, leg_w, leg_h], 
        [bx + dx, by - dy, leg_z], 
        get_color("front_left_leg")
    )
    parts["front_right_leg"] = block(
        [leg_w, leg_w, leg_h], 
        [bx + dx, by + dy, leg_z], 
        get_color("front_right_leg")
    )
    parts["back_left_leg"] = block(
        [leg_w, leg_w, leg_h], 
        [bx - dx, by - dy, leg_z], 
        get_color("back_left_leg")
    )
    parts["back_right_leg"] = block(
        [leg_w, leg_w, leg_h], 
        [bx - dx, by + dy, leg_z], 
        get_color("back_right_leg")
    )

    # -- BACKREST --
    # Attached to the back edge (-X) of the seat
    back_z = bz + leg_h + seat_h + (back_h / 2)
    back_x = bx - (seat_d / 2) + (back_thickness / 2)
    parts["back"] = block(
        [back_thickness, seat_w, back_h], 
        [back_x, by, back_z], 
        get_color("back")
    )

    # -- ARMRESTS --
    # Simple bars connecting back to front at a certain height
    arm_h_offset = 0.25
    arm_len = seat_d
    arm_z = bz + leg_h + seat_h + arm_h_offset
    arm_thick = 0.05
    
    # Left (-Y) and Right (+Y)
    parts["armrest_left"] = block(
        [arm_len, arm_thick, arm_thick],
        [bx, by - (seat_w/2) + (arm_thick/2), arm_z],
        get_color("armrest_left")
    )
    parts["armrest_right"] = block(
        [arm_len, arm_thick, arm_thick],
        [bx, by + (seat_w/2) - (arm_thick/2), arm_z],
        get_color("armrest_right")
    )

    # ---------------------------------------------------------
    # 3. Return Format
    # ---------------------------------------------------------
    # Returns a dictionary: { "part_name": (body_id, link_index) }
    # Since we built separate bodies, link_index is always -1 (Base Link).
    return {name: (bid, -1) for name, bid in parts.items()}