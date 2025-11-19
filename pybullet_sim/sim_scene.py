"""Scene creation utilities used by the simulation demos.

The scene functions are intentionally tiny: create simple geometric
blocks or spawn a prepared chair URDF and return a mapping of named
parts to simulation handles (body id, link index) used by the executor.
"""

import pybullet as p


def block(size, pos, color):
    """Create a simple static box and return its body id.

    Args:
        size: [x, y, z] full extents of the box.
        pos: base position to place the box.
        color: RGBA tuple for the visual shape.
    """
    half = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(0, col, vis, pos)


def spawn_simple_chair(damaged):
    """Load the sample chair URDF and return a name->(body,link) map.

    The function loads `pybullet_sim/assets/chair/chair.urdf` at a fixed
    position and constructs a dictionary mapping friendly part names to
    their corresponding link indices. If `damaged` matches a part name,
    that link's visual color is changed to indicate damage.
    """
    chair = p.loadURDF(
        "pybullet_sim/assets/chair/chair.urdf",
        basePosition=[0.6, 0, 0.4],
        useFixedBase=True
    )

    # Map part names to link indices in the provided URDF. These indices
    # must match the structure of the URDF used by the demo.
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
        # Represent a part as a tuple (body_id, link_index) used elsewhere
        parts[name] = (chair, idx)

        # If this part is the reported damaged one, tint it red.
        if name == damaged:
            p.changeVisualShape(chair, idx, rgbaColor=(1, 0.2, 0.2, 1))

    return parts
