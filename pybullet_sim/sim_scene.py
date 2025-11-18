import pybullet as p

def block(size, pos, color):
    half = [s/2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(0, col, vis, pos)

def spawn_simple_chair(damaged):
    base_x, base_y = 0.6, 0
    parts = {}

    # SEAT - gray
    parts["seat"] = block((0.40,0.40,0.04),(base_x, base_y,0.42),(0.8,0.8,0.8,1))
    
    # BACK - gray or red if damaged
    back_color = (1,0.2,0.2,1) if damaged == "back" else (0.85,0.85,0.85,1)
    parts["back"] = block((0.40,0.06,0.40),(base_x, base_y-0.22,0.64), back_color)

    leg_h = 0.42
    leg_w = 0.06
    leg_positions = {
        "front_left_leg":  (base_x-0.16, base_y+0.16, leg_h/2),
        "front_right_leg": (base_x+0.16, base_y+0.16, leg_h/2),
        "back_left_leg":   (base_x-0.16, base_y-0.16, leg_h/2),
        "back_right_leg":  (base_x+0.16, base_y-0.16, leg_h/2)
    }

    for name,(x,y,z) in leg_positions.items():
        # RED if damaged, GRAY if healthy
        if name == damaged:
            clr = (1,0.2,0.2,1)  # Bright red for damaged
        else:
            clr = (0.7,0.7,0.7,1)  # Gray for healthy
        parts[name] = block((leg_w,leg_w,leg_h),(x,y,z),clr)

    # Add armrests
    armrest_h = 0.08
    armrest_size = (0.08, 0.06, armrest_h)
    armrest_left_color = (1,0.2,0.2,1) if damaged == "armrest_left" else (0.7,0.7,0.7,1)
    armrest_right_color = (1,0.2,0.2,1) if damaged == "armrest_right" else (0.7,0.7,0.7,1)
    
    parts["armrest_left"] = block(armrest_size, (base_x-0.20, base_y, 0.55), armrest_left_color)
    parts["armrest_right"] = block(armrest_size, (base_x+0.20, base_y, 0.55), armrest_right_color)

    # Add front frame
    front_frame_color = (1,0.2,0.2,1) if damaged == "front_frame" else (0.85,0.85,0.85,1)
    parts["front_frame"] = block((0.40,0.06,0.40), (base_x, base_y+0.22, 0.64), front_frame_color)

    return parts
