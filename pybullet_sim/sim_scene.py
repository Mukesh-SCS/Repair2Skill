"""
sim_scene.py — Procedural chair from boxes; dynamic bodies and fixed constraints.
Units: meters, Z-up. Chair center ~ (0.6, 0, 0).
"""

import logging
import pybullet as p

from .sim_connection import get_client

logger = logging.getLogger(__name__)

# Chair layout: part name -> (center_xyz, half_extents_xyz, color_rgba)
# Proportions tuned for a more chair-like look (Option A). All parts are boxes.
# Geometry: seat top 0.43+0.04=0.47; back center 0.66 (back bottom at seat top); leg top 0.19+0.21=0.40 = seat bottom 0.43-0.04=0.39 (leg meets seat).
CHAIR_PARTS = {
    "seat": ((0.6, 0.0, 0.43), (0.22, 0.19, 0.04), (0.55, 0.42, 0.32, 1.0)),
    "back": ((0.6, 0.0, 0.66), (0.22, 0.04, 0.20), (0.5, 0.38, 0.32, 1.0)),
    "front_left_leg": ((0.44, -0.16, 0.19), (0.035, 0.035, 0.21), (0.38, 0.28, 0.20, 1.0)),
    "front_right_leg": ((0.44, 0.16, 0.19), (0.035, 0.035, 0.21), (0.38, 0.28, 0.20, 1.0)),
    "back_left_leg": ((0.76, -0.16, 0.19), (0.035, 0.035, 0.21), (0.42, 0.30, 0.22, 1.0)),
    "back_right_leg": ((0.76, 0.16, 0.19), (0.035, 0.035, 0.21), (0.42, 0.30, 0.22, 1.0)),
    "armrest_left": ((0.6, -0.21, 0.54), (0.21, 0.03, 0.05), (0.52, 0.40, 0.35, 1.0)),
    "armrest_right": ((0.6, 0.21, 0.54), (0.21, 0.03, 0.05), (0.52, 0.40, 0.35, 1.0)),
}

# Non-repairable visual supports (not in CHAIR_PARTS): crossbars under seat. Option A realism.
# List of (center_xyz, half_extents_xyz, color_rgba). Fixed to seat, never removed.
CROSSBARS = [
    ((0.44, 0.0, 0.32), (0.02, 0.035, 0.035), (0.35, 0.26, 0.18, 1.0)),   # front
    ((0.76, 0.0, 0.32), (0.02, 0.035, 0.035), (0.38, 0.28, 0.20, 1.0)),   # back
]

PARENT_MAP = {
    "seat": None,
    "back": "seat",
    "front_left_leg": "seat",
    "front_right_leg": "seat",
    "back_left_leg": "seat",
    "back_right_leg": "seat",
    "armrest_left": "seat",
    "armrest_right": "seat",
}

# Mass per part (kg). Seat is static (0) so the chair base never moves when we remove parts.
PART_MASS = 0.5
SEAT_MASS = 0.0  # static base
# Parts bin: where to spawn replacement parts
PARTS_BIN_POS = (0.9, 0.4, 0.35)


def _create_box(cid, half_extents, pos, orn, mass, color_rgba, friction=(1.0, 0.1, 0.1)):
    """Create a box rigid body. Returns body_id."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color_rgba, physicsClientId=cid)
    body = p.createMultiBody(
        mass,
        col,
        vis,
        pos,
        orn,
        physicsClientId=cid,
    )
    p.changeDynamics(body, -1, lateralFriction=friction[0], spinningFriction=friction[1], rollingFriction=friction[2], physicsClientId=cid)
    return body


class ChairScene:
    """
    Chair built from boxes; each part is a dynamic body connected by fixed constraints.
    Tracks original poses and supports detach / attach / spawn_replacement.
    """

    def __init__(self, chair_center=(0.6, 0.0, 0.0), damaged_part=None):
        self.cid = get_client()
        if self.cid is None:
            raise RuntimeError("PyBullet not connected.")
        self.chair_center = chair_center
        self.damaged_part = damaged_part
        self.bodies = {}
        self.constraints = {}
        self.original_poses = {}
        self._build_chair()

    def _build_chair(self):
        for part_name, (center, half_ext, color) in CHAIR_PARTS.items():
            orn = [0, 0, 0, 1]
            mass = SEAT_MASS if part_name == "seat" else PART_MASS
            body_id = _create_box(
                self.cid,
                half_ext,
                center,
                orn,
                mass=mass,
                color_rgba=color,
            )
            self.bodies[part_name] = body_id
            self.original_poses[part_name] = (list(center), list(orn))
            self.constraints[part_name] = []

        for part_name, parent_name in PARENT_MAP.items():
            if parent_name is None:
                continue
            parent_id = self.bodies[parent_name]
            child_id = self.bodies[part_name]
            pos_c, orn_c = self.original_poses[part_name]
            pos_p, orn_p = self.original_poses[parent_name]
            inv_p, inv_orn_p = p.invertTransform(pos_p, orn_p)
            parent_frame_pos, parent_frame_orn = p.multiplyTransforms(inv_p, inv_orn_p, pos_c, orn_c)
            cid = p.createConstraint(
                parent_id,
                -1,
                child_id,
                -1,
                p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=parent_frame_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=parent_frame_orn,
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self.cid,
            )
            self.constraints[part_name].append(cid)

        # Non-repairable crossbars (visual only, fixed to seat)
        seat_id = self.bodies["seat"]
        orn = [0, 0, 0, 1]
        pos_seat, orn_seat = self.original_poses["seat"]
        for center, half_ext, color in CROSSBARS:
            body_id = _create_box(
                self.cid,
                half_ext,
                center,
                orn,
                mass=0.0,
                color_rgba=color,
            )
            inv_p, inv_orn_p = p.invertTransform(pos_seat, orn_seat)
            parent_frame_pos, parent_frame_orn = p.multiplyTransforms(inv_p, inv_orn_p, center, orn)
            p.createConstraint(
                seat_id, -1, body_id, -1,
                p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=parent_frame_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=parent_frame_orn,
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self.cid,
            )

        if self.damaged_part and self.damaged_part in self.bodies:
            self.recolor(self.damaged_part, (1.0, 0.2, 0.2, 1.0))

    def get_part_pose(self, part):
        """Return (position, orientation) of the current body for part."""
        if part not in self.bodies:
            return None, None
        bid = self.bodies[part]
        pos, orn = p.getBasePositionAndOrientation(bid, physicsClientId=self.cid)
        return list(pos), list(orn)

    def recolor(self, part, rgba):
        """Change visual color of the part."""
        if part not in self.bodies:
            logger.warning("recolor: part %s not found", part)
            return
        bid = self.bodies[part]
        p.changeVisualShape(bid, -1, rgbaColor=rgba, physicsClientId=self.cid)

    def detach(self, part):
        """Break constraint(s) so the part can move freely."""
        if part not in self.constraints:
            logger.warning("detach: part %s not found", part)
            return
        for cid in self.constraints[part]:
            p.removeConstraint(cid, physicsClientId=self.cid)
        self.constraints[part] = []

    def attach(self, part, pose, body_id=None):
        """
        Attach a body to the parent at the given world pose.
        If body_id is None, attach the current body for part (used when re-attaching same body).
        Otherwise attach the given body_id (replacement part) and register it as the part's body.
        """
        parent_name = PARENT_MAP.get(part)
        if parent_name is None:
            if part != "seat":
                logger.warning("attach: part %s has no parent", part)
            return
        parent_id = self.bodies[parent_name]
        if body_id is not None:
            self.bodies[part] = body_id
        child_id = self.bodies[part]
        pos_c, orn_c = pose[0], pose[1]
        pos_p, orn_p = p.getBasePositionAndOrientation(parent_id, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(child_id, pos_c, orn_c, physicsClientId=self.cid)
        inv_p, inv_orn_p = p.invertTransform(pos_p, orn_p)
        parent_frame_pos, parent_frame_orn = p.multiplyTransforms(inv_p, inv_orn_p, pos_c, orn_c)
        cid = p.createConstraint(
            parent_id,
            -1,
            child_id,
            -1,
            p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=parent_frame_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=parent_frame_orn,
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.cid,
        )
        self.constraints[part] = [cid]
        self.original_poses[part] = (list(pos_c), list(orn_c))

    def spawn_replacement(self, part):
        """
        Spawn a new replacement part near the parts bin. Returns body_id.
        Does not register as part's body until attach() is called.
        """
        if part not in CHAIR_PARTS:
            logger.warning("spawn_replacement: part %s not in chair", part)
            return None
        _, half_ext, color = CHAIR_PARTS[part]
        orn = [0, 0, 0, 1]
        body_id = _create_box(
            self.cid,
            half_ext,
            PARTS_BIN_POS,
            orn,
            mass=PART_MASS,
            color_rgba=color,
        )
        return body_id

    def part_body_id(self, part):
        """Return current body id for part, or None."""
        return self.bodies.get(part)
