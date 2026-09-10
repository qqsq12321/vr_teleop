"""Conservative, model-only RealMan self-collision preflight.

The source scene has collision masks disabled. Explicit mj_geomDistance calls
below intentionally bypass those masks; data.ncon==0 is NOT a safety check.
Visual meshes are convex hulls, so false positives are expected. Fingers use
all-articulation bounding spheres because this SDK path has no hand feedback.
This does not detect external obstacles, cables, calibration errors, or provide
continuous swept-volume / certified collision protection.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from utils.core.safety import SafetyFault


class RealManCollisionGuard:
    def __init__(self, scene: str | Path, *, margin: float = 0.02):
        self.margin = float(margin)
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("Positive collision margin required")
        path = Path(scene).resolve()
        source = mujoco.MjModel.from_xml_path(str(path))
        tree = ET.parse(path)
        root = tree.getroot()
        if root.findall("include"):
            raise ValueError("Collision model requires a flattened scene")
        for asset in root.findall("asset/*"):
            if "file" in asset.attrib:
                asset.set("file", str((path.parent / asset.get("file")).resolve()))
        self.hand_radii = {}
        envelope_names = {"left": [], "right": []}
        for side in ("left", "right"):
            palm_name = f"{side}_hand_base_link"
            palm_node = root.find(f".//body[@name='{palm_name}']")
            if palm_node is None:
                raise ValueError(f"Missing {palm_name}")
            # One all-pose sphere at each finger's root is much tighter than
            # one sphere at the wrist. Keep the rigid palm's actual meshes.
            for finger_node in palm_node.findall("body"):
                name = finger_node.get("name")
                bid = source.body(name).id
                radius = 0.0
                for gid in range(source.ngeom):
                    chain = []
                    body = int(source.geom_bodyid[gid])
                    while body != bid and body != 0:
                        chain.append(body)
                        body = int(source.body_parentid[body])
                    if body != bid:
                        continue
                    bound = np.linalg.norm(source.geom_pos[gid]) + source.geom_rbound[gid]
                    for child in chain + [bid]:
                        for jid in range(source.body_jntadr[child], source.body_jntadr[child] + source.body_jntnum[child]):
                            if source.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or np.linalg.norm(source.jnt_pos[jid]) > 1e-9:
                                raise ValueError("Unsupported joint for all-pose finger envelope")
                        if child != bid:
                            bound += np.linalg.norm(source.body_pos[child])
                    radius = max(radius, float(bound))
                if radius <= 0:
                    raise ValueError(f"Missing {name} collision geometry")
                self.hand_radii[name] = radius
                envelope_name = f"{name}_envelope"
                envelope_names[side].append(envelope_name)
                ET.SubElement(finger_node, "geom", {
                    "name": envelope_name, "type": "sphere", "size": str(radius),
                    "contype": "0", "conaffinity": "0", "mass": "0", "group": "3",
                })
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != source.nq or any(self.model.joint(j).name != source.joint(j).name for j in range(source.njnt)):
            raise ValueError("Collision scene changed joint ordering")
        self.groups = {}
        excluded_geoms = set()
        for side, prefix in (("left", "l"), ("right", "r")):
            group = []
            for index in range(1, 8):
                bid = self.model.body(f"{prefix}_link{index}").id
                ids = np.flatnonzero(self.model.geom_bodyid == bid).tolist()
                if not ids:
                    raise ValueError(f"No geometry on {side} link {index}")
                group.extend((gid, index) for gid in ids)
            group.extend((self.model.geom(name).id, 8) for name in envelope_names[side])
            for name in (f"{side}_hand_base_link", f"{side}_arm_end_effector_mount_link"):
                group.extend((int(gid), 8) for gid in np.flatnonzero(self.model.geom_bodyid == self.model.body(name).id))
            self.groups[side] = group
            # All original hand/mount geoms are covered by finger envelopes and rigid palm,
            # not frozen into an arbitrary nominal finger configuration.
            mount = self.model.body(f"{side}_arm_end_effector_mount_link").id
            for gid in range(self.model.ngeom):
                b = int(self.model.geom_bodyid[gid])
                while b and b != mount:
                    b = int(self.model.body_parentid[b])
                if b == mount:
                    excluded_geoms.add(gid)
        moving = {gid for group in self.groups.values() for gid, _ in group}
        fixed = set(range(self.model.ngeom)) - moving - excluded_geoms
        self.pairs = set()
        left, right = self.groups["left"], self.groups["right"]
        self.pairs.update((a, b) for a, _ in left for b, _ in right)
        for side, group in self.groups.items():
            for a, ia in group:
                for b, ib in group:
                    # Neighbouring joints share housings. Finger envelopes
                    # can include their own wrist attachment.
                    if ib > ia + 2 and not (ib == 8 and ia >= 5):
                        self.pairs.add((a, b))
                if ia >= 3:  # Exclude shoulder assembly against its mounting.
                    self.pairs.update((a, b) for b in fixed)
        if not self.pairs:
            raise ValueError("Empty collision pair set")
        self.pairs = sorted(self.pairs)
        self._a = np.array([a for a, _ in self.pairs], dtype=int)
        self._b = np.array([b for _, b in self.pairs], dtype=int)
        self._bounds = self.model.geom_rbound[self._a] + self.model.geom_rbound[self._b]
        # Plane bounds are infinite for broad phase purposes.
        planes = ((self.model.geom_type[self._a] == mujoco.mjtGeom.mjGEOM_PLANE)
                  | (self.model.geom_type[self._b] == mujoco.mjtGeom.mjGEOM_PLANE))
        self._bounds[planes] = np.inf

    def check(self, qpos):
        q = np.asarray(qpos, dtype=float)
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise SafetyFault("Invalid collision model qpos")
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        distances = np.linalg.norm(self.data.geom_xpos[self._a] - self.data.geom_xpos[self._b], axis=1)
        for index in np.flatnonzero(distances <= self._bounds + self.margin):
            a, b = self.pairs[index]
            distance = mujoco.mj_geomDistance(self.model, self.data, a, b, self.margin, None)
            if not np.isfinite(distance) or distance < self.margin:
                names = [self.model.body(int(self.model.geom_bodyid[g])).name for g in (a, b)]
                raise SafetyFault(f"Collision clearance: {names[0]} / {names[1]} ({distance:.4f} m)")

    def check_transition(self, measured, previous, candidate):
        # Serial left/right writes are NOT atomic: check each unilateral pose
        # as well as the combined target and sampled simultaneous path.
        self.check(measured)
        self.check(previous)
        changed = np.flatnonzero(np.asarray(candidate) != np.asarray(previous))
        for prefix in ("l", "r"):
            unilateral = np.array(previous, copy=True)
            addresses = [self.model.jnt_qposadr[self.model.joint(f"{prefix}_joint{i}").id] for i in range(1, 8)]
            unilateral[addresses] = np.asarray(candidate)[addresses]
            self.check(unilateral)
        if changed.size:
            for fraction in (0.5, 1.0):
                self.check(np.asarray(previous) + fraction * (np.asarray(candidate) - previous))
                self.check(np.asarray(measured) + fraction * (np.asarray(candidate) - measured))
