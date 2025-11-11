import pybullet as p

def create_block(size, pos, color):
    half = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(0, col, vis, pos)

def spawn_simple_chair(damaged_part="back_left_leg"):
    base_x, base_y = 0.6, 0.0
    parts = {}
    parts["seat"] = create_block((0.4, 0.4, 0.04), (base_x, base_y, 0.42), (0.8, 0.8, 0.8, 1))
    parts["back"] = create_block((0.4, 0.06, 0.4), (base_x, base_y - 0.22, 0.64), (0.85, 0.85, 0.85, 1))
    leg_h, leg_w = 0.42, 0.06
    legs = {
        "front_left_leg":  (base_x - 0.16, base_y + 0.16, leg_h / 2),
        "front_right_leg": (base_x + 0.16, base_y + 0.16, leg_h / 2),
        "back_left_leg":   (base_x - 0.16, base_y - 0.16, leg_h / 2),
        "back_right_leg":  (base_x + 0.16, base_y - 0.16, leg_h / 2)
    }
    for name, (x, y, z) in legs.items():
        color = (1, 0.3, 0.3, 1) if name == damaged_part else (0.7, 0.7, 0.7, 1)
        parts[name] = create_block((leg_w, leg_w, leg_h), (x, y, z), color)
    return parts
