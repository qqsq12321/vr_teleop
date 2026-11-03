"""Capsule self-collision model for IK avoidance and a hard clearance check.

Each link gets one capsule fitted to its own meshes. A hand gets two envelope
capsules around its geometry at the rest (open) pose: palm plus four fingers,
and the thumb. One capsule for the whole hand degenerates into a sphere, since
an open thumb sticks out sideways further than half the hand's length. Capsule-capsule
distance is the closest-segment distance minus both radii, which is analytic,
cheap and smooth, so it can run inside every IK iteration and supply a
gradient that steers the arm away from itself rather than just stopping it.

The capsules are conservative: on the RM75-B the meshes fill only about a third
to a half of their fitted capsules, mostly because the joint housings bulge
out sideways. Expect avoidance to start a few centimetres before metal would
touch.

An optional floor is a horizontal plane every moving capsule must stay above,
held off with the same influence, margin and soft rows as the self pairs.
Nothing else is modelled -- not a table, objects or cables -- and this is no
substitute for the controller's own collision detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Pairs closer than this along the kinematic chain share a joint housing, so
# their capsules overlap by construction and are never checked.
MIN_CHAIN_GAP = 3
# Fingers curl toward the palm when closing, which can poke slightly outside
# the envelope fitted to the open hand. The thumb also swings about its base,
# so it gets more room.
HAND_ENVELOPE_PAD_M = 0.01
THUMB_ENVELOPE_PAD_M = 0.02
# Name reported for the floor in place of a second capsule.
FLOOR = "floor"


@dataclass(frozen=True)
class Capsule:
    body_id: int
    name: str
    # Position along the kinematic chain; capsules on one rigid body share it.
    level: int
    # Segment end points and radius, in the body frame.
    p0: np.ndarray
    p1: np.ndarray
    radius: float


def _mesh_points_in_frame(model, data, geom_ids, frame_body):
    """Mesh vertices of ``geom_ids``, expressed in ``frame_body``'s frame."""
    origin = data.xpos[frame_body]
    rotation = data.xmat[frame_body].reshape(3, 3)
    points = []
    for gid in geom_ids:
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[gid]
        start = model.mesh_vertadr[mesh]
        verts = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        world = verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        points.append((world - origin) @ rotation)
    if not points:
        raise ValueError("No mesh geometry to fit a capsule to")
    return np.vstack(points)


def fit_capsule(points, pad=0.0):
    """Smallest-axis capsule around ``points`` along their principal direction.

    The radius is the largest point-to-segment distance, so every point is
    inside by construction -- including near the hemispherical caps.
    """
    centre = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centre, full_matrices=False)
    axis = vt[0]
    t = (points - centre) @ axis
    radial = np.linalg.norm((points - centre) - np.outer(t, axis), axis=1).max()
    t0, t1 = t.min() + radial, t.max() - radial
    if t1 < t0:
        t0 = t1 = 0.5 * (t.min() + t.max())
    p0, p1 = centre + t0 * axis, centre + t1 * axis
    radius = _point_segment_distances(points, p0, p1).max()
    return p0, p1, float(radius + pad)


def _point_segment_distances(points, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-18:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + np.outer(t, ab)), axis=1)


def closest_points(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2 (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            s, t = np.clip(-c / a, 0.0, 1.0), 0.0
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = np.clip(-c / a, 0.0, 1.0), 0.0
            elif t > 1.0:
                s, t = np.clip((b - c) / a, 0.0, 1.0), 1.0
    return p1 + d1 * s, p2 + d2 * t


class CapsuleSet:
    """Self-collision capsules for one serial arm, plus an optional hand.

    ``chain`` lists the link bodies from base to flange; ``hand`` names the
    hand's root body, whose subtree becomes the palm and thumb envelopes.
    ``floor_z`` is the world height of the floor, or None for no floor. The
    first chain link stands on the floor and is not checked against it.
    """

    def __init__(self, model, chain, *, hand=None, influence=0.05, margin=0.005,
                 weight=5.0, rest_qpos=None, floor_z=None):
        if not 0.0 <= margin < influence:
            raise ValueError("Need 0 <= margin < influence")
        self.model = model
        self.influence = float(influence)
        self.margin = float(margin)
        self.weight = float(weight)
        data = mujoco.MjData(model)
        if rest_qpos is not None:
            data.qpos[:] = rest_qpos
        mujoco.mj_kinematics(model, data)

        capsules = []
        for level, name in enumerate(chain):
            body = model.body(name).id
            geoms = np.flatnonzero(model.geom_bodyid == body)
            capsules.append(Capsule(body, name, level, *fit_capsule(
                _mesh_points_in_frame(model, data, geoms, body))))
        if hand is not None:
            root = model.body(hand).id
            subtree = [b for b in range(model.nbody) if self._descends(b, root)]
            thumb = [b for b in subtree if "thumb" in model.body(b).name]
            parts = [("palm", [b for b in subtree if b not in thumb], HAND_ENVELOPE_PAD_M)]
            if thumb:
                parts.append(("thumb", thumb, THUMB_ENVELOPE_PAD_M))
            # Both envelopes are rigid on the hand root: the real-robot loop
            # never updates finger joints in the model, so attaching them to
            # moving finger bodies would not track the real hand anyway.
            for label, bodies, pad in parts:
                geoms = np.flatnonzero(np.isin(model.geom_bodyid, bodies))
                capsules.append(Capsule(root, f"{hand}:{label}", len(chain), *fit_capsule(
                    _mesh_points_in_frame(model, data, geoms, root), pad=pad)))
        self.capsules = capsules
        self.pairs = [
            (i, j)
            for i in range(len(capsules)) for j in range(i + 1, len(capsules))
            if abs(capsules[j].level - capsules[i].level) >= MIN_CHAIN_GAP
        ]
        self._jacp = np.zeros((3, model.nv))
        # Broad phase: per-capsule bounding sphere about the segment midpoint.
        self._bodies = np.array([c.body_id for c in capsules])
        self._local = np.array([[c.p0, c.p1] for c in capsules])  # (n, 2, 3)
        self._bound = np.array(
            [0.5 * np.linalg.norm(c.p1 - c.p0) + c.radius for c in capsules]
        )
        self._pair_i = np.array([i for i, _ in self.pairs], dtype=int)
        self._pair_j = np.array([j for _, j in self.pairs], dtype=int)
        self.floor_z = None if floor_z is None else float(floor_z)
        self.floor_checked = (
            [] if floor_z is None else [i for i, c in enumerate(capsules) if c.level > 0]
        )

    def _descends(self, body, root):
        while body != 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return root == 0

    def _world_segments(self, data):
        rotation = data.xmat[self._bodies].reshape(-1, 3, 3)
        origin = data.xpos[self._bodies]
        # (n, 2, 3): both end points of every capsule in world coordinates.
        return np.einsum("nij,nkj->nki", rotation, self._local) + origin[:, None, :]

    def distances(self, data, cutoff=np.inf):
        """Yield (pair_index, distance, point_a, point_b) for pairs that may be
        closer than ``cutoff``.

        A vectorised bounding-sphere test discards pairs that cannot be within
        ``cutoff``; only the rest get the exact segment-segment computation.
        Inside the IK, most pairs are far apart, so this skips most of the work.
        """
        segments = self._world_segments(data)
        if np.isfinite(cutoff):
            centres = segments.mean(axis=1)
            lower = (
                np.linalg.norm(centres[self._pair_i] - centres[self._pair_j], axis=1)
                - self._bound[self._pair_i] - self._bound[self._pair_j]
            )
            candidates = np.flatnonzero(lower < cutoff)
        else:
            candidates = range(len(self.pairs))
        for index in candidates:
            i, j = self.pairs[index]
            pa, pb = closest_points(segments[i, 0], segments[i, 1], segments[j, 0], segments[j, 1])
            gap = float(np.linalg.norm(pa - pb))
            yield int(index), gap - self.capsules[i].radius - self.capsules[j].radius, pa, pb

    def floor_distances(self, data):
        """Yield (capsule_index, distance, point_on_axis, point_on_floor) for
        every capsule checked against the floor; nothing without a floor.

        The axis point is the segment's lower end, which is the closest point
        to a horizontal plane.
        """
        if self.floor_z is None:
            return
        segments = self._world_segments(data)
        for i in self.floor_checked:
            p0, p1 = segments[i]
            low = p0 if p0[2] <= p1[2] else p1
            yield (i, float(low[2] - self.floor_z - self.capsules[i].radius),
                   low, np.array([low[0], low[1], self.floor_z]))

    def min_clearance(self, data):
        """Smallest surface-to-surface distance and the two names involved
        (the second is ``FLOOR`` when the floor is closest)."""
        best, names = np.inf, None
        for index, distance, _pa, _pb in self.distances(data):
            if distance < best:
                i, j = self.pairs[index]
                best, names = distance, (self.capsules[i].name, self.capsules[j].name)
        for i, distance, _pa, _pb in self.floor_distances(data):
            if distance < best:
                best, names = distance, (self.capsules[i].name, FLOOR)
        return best, names

    def clearance_per_capsule(self, data):
        """Per capsule: the worst clearance to anything it is checked against."""
        worst = np.full(len(self.capsules), np.inf)
        for index, distance, _pa, _pb in self.distances(data):
            i, j = self.pairs[index]
            worst[i] = min(worst[i], distance)
            worst[j] = min(worst[j], distance)
        for i, distance, _pa, _pb in self.floor_distances(data):
            worst[i] = min(worst[i], distance)
        return worst

    def constraint_rows(self, model, data):
        """Soft IK rows pushing every pair inside ``influence`` back out to it.

        Each active pair is a progressive spring on its gap ``d``: with
        ``s = (influence - d) / (influence - margin)``, the residual is
        ``weight * (influence - margin) * s**2`` and its Jacobian row is
        ``2 * weight * s * n^T (J_a - J_b)``. Both vanish at the influence
        boundary, so the push fades in with no kink and the arm yields softly
        at first, then stiffens to the full ``weight`` by ``margin`` and keeps
        stiffening past it. ``data`` must hold current kinematics.
        """
        errors, rows = [], []
        for index, distance, pa, pb in self.distances(data, cutoff=self.influence):
            if distance >= self.influence:
                continue
            i, j = self.pairs[index]
            delta = pa - pb
            norm = float(np.linalg.norm(delta))
            if norm < 1e-9:
                continue  # Coincident axes: no defined push direction.
            normal = delta / norm
            row = np.zeros(model.nv)
            for body, point, sign in ((self.capsules[i].body_id, pa, 1.0),
                                      (self.capsules[j].body_id, pb, -1.0)):
                self._jacp[:] = 0.0
                mujoco.mj_jac(model, data, self._jacp, None, point, body)
                row += sign * (normal @ self._jacp)
            error, gain = self._spring(distance)
            errors.append(error)
            rows.append(gain * row)
        for i, distance, pa, _pb in self.floor_distances(data):
            if distance >= self.influence:
                continue
            # The floor does not move and its normal is always up, so only the
            # capsule's vertical velocity at its lowest point opens the gap.
            self._jacp[:] = 0.0
            mujoco.mj_jac(model, data, self._jacp, None, pa, self.capsules[i].body_id)
            error, gain = self._spring(distance)
            errors.append(error)
            rows.append(gain * self._jacp[2])
        if not errors:
            return np.zeros(0), np.zeros((0, model.nv))
        return np.asarray(errors), np.vstack(rows)

    def _spring(self, distance):
        """Residual and row gain of the progressive spring at gap ``distance``."""
        span = self.influence - self.margin
        s = (self.influence - distance) / span
        return self.weight * span * s * s, 2.0 * self.weight * s


_GREEN = np.array([0.2, 0.8, 0.3, 0.25], dtype=np.float32)
_ORANGE = np.array([1.0, 0.6, 0.1, 0.5], dtype=np.float32)
_RED = np.array([1.0, 0.1, 0.1, 0.7], dtype=np.float32)


def draw_capsules(scene, capsule_set, data):
    """Overlay ``capsule_set`` on a viewer's ``user_scn``, coloured by clearance.

    Each capsule takes its worst clearance to a partner or the floor. Green is
    clear of ``influence``, orange is inside it (the IK is pushing the
    pair apart), red is inside ``margin`` (the real robot would hold the frame).
    Replaces whatever was in ``scene``. Returns the smallest clearance.
    """
    worst = capsule_set.clearance_per_capsule(data)
    scene.ngeom = 0
    for capsule, (p0, p1), clearance in zip(
        capsule_set.capsules, capsule_set._world_segments(data), worst
    ):
        if scene.ngeom >= scene.maxgeom:
            break
        color = (_RED if clearance < capsule_set.margin
                 else _ORANGE if clearance < capsule_set.influence else _GREEN)
        geom = scene.geoms[scene.ngeom]
        if np.linalg.norm(p1 - p0) < 1e-6:
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([capsule.radius, 0, 0]), p0, np.eye(3).flatten(), color)
        else:
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).flatten(), color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, capsule.radius, p0, p1)
        scene.ngeom += 1
    return float(worst.min())


__all__ = ["FLOOR", "Capsule", "CapsuleSet", "closest_points", "draw_capsules", "fit_capsule"]
