"""Scene creation utilities used by the simulation demos.

The scene functions are intentionally tiny: create simple geometric
blocks or spawn a procedural chair and return a mapping of named
parts to simulation handles (body id, link index) used by the executor.
"""

import pybullet as p

def block(size, pos, color, mass=0):
    """Create a box and return its body id.

    Args:
        size: [x, y, z] full extents of the box.
        pos: base position to place the box (center of mass).
        color: RGBA tuple for the visual shape.
        mass: Mass of the body. 0 = static (default), >0 = dynamic (movable).
    """
    half = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)

    return p.createMultiBody(mass, col, vis, pos)


def make_part_dynamic(body_id, mass=1.0):
    """Make a static part dynamic by changing its mass.
    
    This allows a previously static part to be picked up and moved.
    PyBullet doesn't allow changing mass directly, so we recreate the body.
    
    Args:
        body_id: The PyBullet body ID to make dynamic
        mass: The mass to give the body (default 1.0 kg)
        
    Returns:
        int: New body ID (the old body is removed)
    """
    # Get the current state of the body
    pos, orn = p.getBasePositionAndOrientation(body_id)
    
    # Get visual shape info
    visual_data = p.getVisualShapeData(body_id)
    if not visual_data:
        print(f"[WARNING] Could not get visual data for body {body_id}")
        return body_id
    
    # Extract dimensions and color from visual data
    # visual_data format: (bodyId, linkIndex, visualGeometryType, dimensions, meshFileName, localVisualPos, localVisualOrn, rgbaColor)
    vis_info = visual_data[0]
    geom_type = vis_info[2]
    dimensions = vis_info[3]  # For box: halfExtents
    color = vis_info[7]
    
    # Get collision shape info
    # We'll recreate the collision shape based on visual dimensions
    if geom_type == p.GEOM_BOX:
        half_extents = dimensions  # Already half extents for boxes
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    else:
        print(f"[WARNING] Unsupported geometry type {geom_type} for body {body_id}")
        return body_id
    
    # Remove old body
    p.removeBody(body_id)
    
    # Create new dynamic body at same position
    new_body_id = p.createMultiBody(mass, col, vis, pos, orn)
    
    # Set realistic dynamics with damping for stable grasping
    p.changeDynamics(new_body_id, -1, 
                    lateralFriction=1.0,
                    spinningFriction=0.2,
                    rollingFriction=0.1,
                    linearDamping=0.05,
                    angularDamping=0.05)
    
    print(f"[SCENE] Made body dynamic: old_id={body_id} -> new_id={new_body_id}, mass={mass}")
    
    return new_body_id


def create_dynamic_block(size, pos, color, mass=1.0):
    """Create a dynamic (movable) box that can be picked up.
    
    Args:
        size: [x, y, z] full extents of the box.
        pos: base position to place the box.
        color: RGBA tuple for the visual shape.
        mass: Mass in kg (default 1.0).
        
    Returns:
        int: Body ID of the created block
    """
    half = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    
    body_id = p.createMultiBody(mass, col, vis, pos)
    
    # Set realistic dynamics with damping for stable grasping
    p.changeDynamics(body_id, -1,
                    lateralFriction=1.0,
                    spinningFriction=0.2,
                    rollingFriction=0.1,
                    linearDamping=0.05,
                    angularDamping=0.05)
    
    return body_id


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
   
    bx, by, bz = 0.5, 0.0, 0.0  # Chair base position (closer to robot for reach)
    
    # =========================================================================
    # REAL-WORLD SCALE CHAIR DIMENSIONS (meters)
    # =========================================================================
    # A realistic chair that matches the KUKA robot's real-world scale.
    # KUKA iiwa is ~1.3m tall, so the chair should be ~0.9m tall.
    # 
    # This fixes the "robot looks gigantic" problem - the issue was the
    # chair was toy-scale (20cm) while the robot was real-scale (1.3m).
    # =========================================================================
    
    # Seat dimensions (realistic office chair)
    seat_w = 0.45    # 45 cm width
    seat_d = 0.45    # 45 cm depth
    seat_h = 0.04    # 4 cm thick
    
    # Leg dimensions
    leg_w = 0.04     # 4 cm square legs
    leg_h = 0.45     # 45 cm tall (seat at 45cm height)
    
    # Backrest dimensions (grippable by industrial gripper)
    back_h = 0.50          # 50 cm tall backrest
    back_thickness = 0.04  # 4 cm thick - fits between gripper fingers
    back_width = 0.40      # 40 cm wide
    
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

    # -- BACKREST -- (sized to fit between gripper fingers)
   
    back_z = bz + leg_h + seat_h + (back_h / 2)
    back_x = bx - (seat_d / 2) + (back_thickness / 2)
    parts["back"] = block(
        [back_thickness, back_width, back_h],  # Thin and narrow for gripping
        [back_x, by, back_z], 
        get_color("back")
    )

    # -- ARMRESTS -- (real-world scale, grippable)
   
    arm_h_offset = 0.20  # Armrests 20cm above seat
    arm_len = seat_d * 0.8  # 36cm long
    arm_z = bz + leg_h + seat_h + arm_h_offset
    arm_thick = 0.04  # 4cm thick - fits between gripper fingers
    
   
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