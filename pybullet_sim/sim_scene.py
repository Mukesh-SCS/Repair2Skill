import pybullet as p

def block(size, pos, color):
    half = [s/2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(0, col, vis, pos)

def spawn_simple_chair(damaged):
    chair = p.loadURDF(
        "pybullet_sim/assets/chair/chair.urdf",
        basePosition=[0.6, 0, 0.4],
        useFixedBase=True
    )

    # Map part names to link indices
    name_to_link = {
        "seat": 0,
        "back": 1,
        "front_left_leg": 2,
        "front_right_leg": 3,
        "back_left_leg": 4,
        "back_right_leg": 5,
        "armrest_left": 6,
        "armrest_right": 7
    }

    parts = {}
    for name, idx in name_to_link.items():
        parts[name] = (chair, idx)

        if name == damaged:
            p.changeVisualShape(chair, idx, rgbaColor=(1, 0.2, 0.2, 1))

    return parts
