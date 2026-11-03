"""Capsule self- and floor-collision model and its use in IK, on the single-arm RM75-B."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from utils.core.body.ik import solve_body_pose_ik, solve_pose_ik
from utils.core.capsule_collision import (
    FLOOR,
    CapsuleSet,
    _mesh_points_in_frame,
    closest_points,
    fit_capsule,
)
from utils.core.quaternion import matrix_to_quaternion
from utils.robots.realman_rm75b_single import (
    RealManRM75BSingleInspireSpec,
    RealManRM75BSingleSpec,
)

ROOT = Path(__file__).resolve().parents[1]
ARM = np.arange(7)
# Drags the flange back toward the base and upper arm, which folds the forearm
# and hand into the arm without any avoidance.
FOLD_GOAL = np.array([0.05, 0.0, 0.35])
# Out in front and below the floor: drives the hand down into it.
DIVE_GOAL = np.array([0.5, 0.0, -0.1])

_ENTRY = importlib.util.spec_from_file_location(
    "teleop_rm75b_single_real", ROOT / "example" / "teleop_rm75b_single_real.py"
)
entry = importlib.util.module_from_spec(_ENTRY)
_ENTRY.loader.exec_module(entry)


@pytest.fixture(scope="module", params=["none", "inspire"])
def rig(request):
    spec = {"none": RealManRM75BSingleSpec, "inspire": RealManRM75BSingleInspireSpec}[request.param]()
    model = mujoco.MjModel.from_xml_path(str(spec.scene_path))
    capsules = CapsuleSet(model, spec.collision_chain, hand=spec.collision_hand,
                          floor_z=spec.collision_floor_z)
    return spec, model, capsules


def _home(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _clearance(model, capsules, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return capsules.min_clearance(data)[0]


def _floor_clearance(model, capsules, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return min(d for _i, d, _a, _b in capsules.floor_distances(data))


def _fold(model, capsules, *, avoid, steps=150, goal=FOLD_GOAL):
    """Teleop-style: walk the target toward ``goal`` one small step a frame."""
    data = _home(model)
    site = model.site("right_tcp").id
    start = data.site_xpos[site].copy()
    quat = matrix_to_quaternion(data.site_xmat[site].reshape(3, 3))
    q, workspace, worst = data.qpos.copy(), mujoco.MjData(model), np.inf
    for k in range(steps):
        target = start + (goal - start) * min(1.0, k / 100)
        q = solve_pose_ik(
            model, workspace, site, target, quat, q, home_qpos=q[:7], skip_tail_joints=0,
            dof_indices=ARM, current_q_weight=0.3,
            collision=capsules if avoid else None, joint_limit_margin=0.05 if avoid else None,
        )
        worst = min(worst, _clearance(model, capsules, q))
    return q, worst


# -- geometry ----------------------------------------------------------------


@pytest.mark.parametrize(
    "segments, expected",
    [
        # Crossing at right angles, 1 apart.
        ((np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]),
          np.array([0, -1.0, 1]), np.array([0, 1.0, 1])), 1.0),
        # Parallel, offset by 2.
        ((np.array([0.0, 0, 0]), np.array([1.0, 0, 0]),
          np.array([0, 2.0, 0]), np.array([1.0, 2.0, 0])), 2.0),
        # End to end along one line.
        ((np.array([0.0, 0, 0]), np.array([1.0, 0, 0]),
          np.array([3.0, 0, 0]), np.array([4.0, 0, 0])), 2.0),
        # Degenerate segment (a point) against a segment.
        ((np.array([0.5, 1.0, 0]), np.array([0.5, 1.0, 0]),
          np.array([0.0, 0, 0]), np.array([1.0, 0, 0])), 1.0),
    ],
)
def test_closest_points_between_segments(segments, expected):
    pa, pb = closest_points(*segments)
    assert np.linalg.norm(pa - pb) == pytest.approx(expected)


def test_fitted_capsule_contains_every_point():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3)) * np.array([0.2, 0.03, 0.05])
    p0, p1, radius = fit_capsule(points)
    ab = p1 - p0
    t = np.clip((points - p0) @ ab / (ab @ ab), 0.0, 1.0)
    assert np.all(np.linalg.norm(points - (p0 + np.outer(t, ab)), axis=1) <= radius + 1e-12)


def test_inspire_hand_is_two_envelopes_not_one_sphere(rig):
    """A single envelope degenerates into a sphere around the open thumb."""
    spec, _model, capsules = rig
    hand = [c for c in capsules.capsules if c.name.startswith(f"{spec.collision_hand}:")]
    if spec.collision_hand is None:
        assert hand == []
        return
    assert sorted(c.name.split(":")[1] for c in hand) == ["palm", "thumb"]
    palm = next(c for c in hand if c.name.endswith(":palm"))
    assert np.linalg.norm(palm.p1 - palm.p0) > 0.1  # Elongated along the fingers.
    # Both envelopes ride on one rigid body, so they are never paired.
    levels = {c.level for c in hand}
    assert len(levels) == 1


def test_adjacent_links_are_never_paired(rig):
    _spec, _model, capsules = rig
    for i, j in capsules.pairs:
        assert abs(capsules.capsules[i].level - capsules.capsules[j].level) >= 3


# Worst case of each skipped level-gap-2 pair, found by sweeping the only two
# joints that move them relative to each other over their full range at 2.5 deg
# (joint_1..joint_7 in degrees, the rest zero). Guards the reason they are
# skipped: the joint limits already keep them apart, so no capsule is needed.
WORST_GAP_TWO_POSES = (
    ("base_link", "link_2", (40, -130, 0, 0, 0, 0, 0)),
    ("link_1", "link_3", (0, -130, -138, 0, 0, 0, 0)),
    ("link_2", "link_4", (0, 0, -178, 135, 0, 0, 0)),
    ("link_3", "link_5", (0, 0, 0, 135, -74, 0, 0)),
    ("link_4", "link_6", (0, 0, 0, 0, 15, -128, 0)),
    ("link_5", "link_7", (0, 0, 0, 0, 0, 128, -10)),
)


def _mesh_in_world(model, data, body_name):
    body = model.body(body_name).id
    geoms = np.flatnonzero(model.geom_bodyid == body)
    local = _mesh_points_in_frame(model, data, geoms, body)
    return local @ data.xmat[body].reshape(3, 3).T + data.xpos[body]


def _mesh_gap(model, data, name_a, name_b):
    """Closest distance between two bodies' mesh vertices, in world space.

    Vertex to vertex, so it can overstate the true surface gap by up to the
    vertex spacing (median 1.5-2 mm on these meshes, 95th percentile ~5 mm).
    """
    a, b = _mesh_in_world(model, data, name_a), _mesh_in_world(model, data, name_b)
    best = np.inf
    for chunk in np.array_split(a, 1 + len(a) // 256):
        best = min(best, float(np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2).min()))
    return best


@pytest.mark.parametrize("name_a, name_b, degrees", WORST_GAP_TWO_POSES)
def test_skipped_gap_two_pairs_cannot_reach_each_other(name_a, name_b, degrees):
    model = mujoco.MjModel.from_xml_path(str(RealManRM75BSingleSpec().scene_path))
    data = mujoco.MjData(model)
    data.qpos[:7] = np.deg2rad(degrees)
    mujoco.mj_forward(model, data)
    # Measured 17-58 mm; 10 mm leaves room for the vertex-spacing error above.
    assert _mesh_gap(model, data, name_a, name_b) > 0.010


def test_home_and_zero_poses_are_clear(rig):
    """Conservative capsules must not already overlap where the arm starts."""
    _spec, model, capsules = rig
    assert _clearance(model, capsules, _home(model).qpos) > capsules.influence
    assert _clearance(model, capsules, np.zeros(model.nq)) > capsules.influence


def test_broad_phase_never_drops_a_close_pair(rig):
    _spec, model, capsules = rig
    q, _worst = _fold(model, capsules, avoid=False, steps=110)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    exhaustive = {i: d for i, d, _a, _b in capsules.distances(data)}
    filtered = {i: d for i, d, _a, _b in capsules.distances(data, cutoff=capsules.influence)}
    for index, distance in exhaustive.items():
        if distance < capsules.influence:
            assert filtered[index] == pytest.approx(distance)


def test_distance_gradient_matches_finite_differences(rig):
    _spec, model, capsules = rig
    q, _worst = _fold(model, capsules, avoid=True, steps=90)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    errors, rows = capsules.constraint_rows(model, data)
    assert errors.size  # The fold leaves at least one pair inside influence.
    # err = w * (influence - d), so d(err)/dq = -w * dd/dq and row = w * dd/dq.
    base = errors[0]
    eps = 1e-6
    for k in ARM:
        bumped = mujoco.MjData(model)
        bumped.qpos[:] = q
        bumped.qpos[k] += eps
        mujoco.mj_forward(model, bumped)
        err_k, _rows = capsules.constraint_rows(model, bumped)
        assert -(err_k[0] - base) / eps == pytest.approx(rows[0, k], abs=1e-4)


def test_floor_is_off_unless_asked():
    spec = RealManRM75BSingleSpec()
    model = mujoco.MjModel.from_xml_path(str(spec.scene_path))
    capsules = CapsuleSet(model, spec.collision_chain)
    assert capsules.floor_z is None
    assert list(capsules.floor_distances(_home(model))) == []


def test_every_moving_capsule_is_kept_off_the_floor(rig):
    """The base stands on the floor; everything above it is checked."""
    _spec, _model, capsules = rig
    names = [capsules.capsules[i].name for i in capsules.floor_checked]
    assert "base_link" not in names
    assert len(names) == len(capsules.capsules) - 1


def test_floor_gradient_matches_finite_differences(rig):
    _spec, model, capsules = rig
    q, _worst = _fold(model, capsules, avoid=True, steps=150, goal=DIVE_GOAL)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    assert _floor_clearance(model, capsules, q) < capsules.influence  # A floor row is active.
    errors, rows = capsules.constraint_rows(model, data)
    eps = 1e-6
    for k in ARM:
        bumped = mujoco.MjData(model)
        bumped.qpos[:] = q
        bumped.qpos[k] += eps
        mujoco.mj_forward(model, bumped)
        err_k, _rows = capsules.constraint_rows(model, bumped)
        # Floor rows come last.
        assert -(err_k[-1] - errors[-1]) / eps == pytest.approx(rows[-1, k], abs=1e-4)


# -- IK ----------------------------------------------------------------------


def test_avoidance_keeps_the_arm_out_of_itself(rig):
    _spec, model, capsules = rig
    _q, unprotected = _fold(model, capsules, avoid=False)
    q, protected = _fold(model, capsules, avoid=True)
    assert unprotected < 0.0  # Without avoidance the fold runs into itself.
    assert protected > capsules.margin
    lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    assert np.all(q[:7] >= lower) and np.all(q[:7] <= upper)


def test_avoidance_keeps_the_hand_off_the_floor(rig):
    _spec, model, capsules = rig
    q_off, _worst = _fold(model, capsules, avoid=False, goal=DIVE_GOAL)
    q_on, worst = _fold(model, capsules, avoid=True, goal=DIVE_GOAL)
    assert _floor_clearance(model, capsules, q_off) < 0.0  # Unprotected, it goes through.
    assert worst > capsules.margin


def test_joint_limit_margin_holds_even_without_collision():
    model = mujoco.MjModel.from_xml_path(str(RealManRM75BSingleSpec().scene_path))
    data = _home(model)
    site = model.site("right_tcp").id
    quat = matrix_to_quaternion(data.site_xmat[site].reshape(3, 3))
    far = data.site_xpos[site] + np.array([0.0, 0.0, -1.0])  # Unreachable.
    q = solve_pose_ik(model, mujoco.MjData(model), site, far, quat, data.qpos.copy(),
                      home_qpos=data.qpos[:7], skip_tail_joints=0, dof_indices=ARM,
                      joint_limit_margin=0.05)
    lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    assert np.all(q[:7] >= lower + 0.05 - 1e-12) and np.all(q[:7] <= upper - 0.05 + 1e-12)


def test_body_ik_takes_the_same_options(rig):
    """The sim's body-target IK gets the same opt-in avoidance."""
    _spec, model, capsules = rig
    data = _home(model)
    body = model.body("right_arm_end_effector_mount_link").id
    quat = matrix_to_quaternion(data.xmat[body].reshape(3, 3))
    q, workspace = data.qpos.copy(), mujoco.MjData(model)
    start = data.xpos[body].copy()
    for k in range(150):
        q = solve_body_pose_ik(
            model, workspace, body, start + (FOLD_GOAL - start) * min(1.0, k / 100), quat, q,
            home_qpos=q[:7], dof_indices=ARM, collision=capsules, joint_limit_margin=0.05,
        )
    assert _clearance(model, capsules, q) > capsules.margin


# -- hard backstop and wiring --------------------------------------------------


class _Clock:
    now = 0.0

    def monotonic(self):
        return self.now


def test_backstop_passes_a_clear_step_unchanged(rig):
    spec, model, _capsules = rig
    backstop = entry._CollisionBackstop(model, spec, emit=lambda _m: None)
    home = _home(model).qpos.copy()
    step = home.copy()
    step[0] += 0.05
    assert backstop.limit(home, step) is step


def test_backstop_stops_a_step_at_the_margin_instead_of_dropping_it(rig):
    spec, model, _capsules = rig
    said = []
    backstop = entry._CollisionBackstop(model, spec, emit=said.append, clock=_Clock())
    home = _home(model).qpos.copy()
    colliding, _worst = _fold(model, backstop.capsules, avoid=False)
    limited = backstop.limit(home, colliding)
    assert limited is not None
    clearance = backstop.clearance(limited)[0]
    # It travels right up to the margin: safe, but not short by much.
    assert backstop.capsules.margin <= clearance < backstop.capsules.margin + 0.01
    assert np.linalg.norm(limited - home) > 0.5 * np.linalg.norm(colliding - home)
    assert said[0].startswith("Limiting:")


def test_backstop_barely_moves_once_at_the_margin(rig):
    """Pushing deeper from the margin creeps along it at most -- never past."""
    spec, model, _capsules = rig
    backstop = entry._CollisionBackstop(model, spec, emit=lambda _m: None)
    colliding, _worst = _fold(model, backstop.capsules, avoid=False)
    q = backstop.limit(_home(model).qpos.copy(), colliding)
    for _ in range(5):
        nxt = backstop.limit(q, colliding)
        if nxt is None:
            break
        assert np.linalg.norm(nxt - q) < 0.01 * np.linalg.norm(colliding - q)
        assert backstop.clearance(nxt)[0] >= backstop.capsules.margin
        q = nxt


def _inside_and_deeper(model, backstop):
    """Two poses inside the margin, the second further in than the first."""
    colliding, _worst = _fold(model, backstop.capsules, avoid=False)
    # Along this path from home the clearance falls monotonically into the fold.
    home = _home(model).qpos.copy()
    shallow, deep = colliding + 0.4 * (home - colliding), colliding + 0.3 * (home - colliding)
    margin = backstop.capsules.margin
    assert backstop.clearance(deep)[0] < backstop.clearance(shallow)[0] < margin
    return shallow, deep


def test_backstop_holds_inside_the_margin_and_warns_once_a_second(rig):
    spec, model, _capsules = rig
    said = []
    clock = _Clock()
    backstop = entry._CollisionBackstop(model, spec, emit=said.append, clock=clock)
    shallow, deep = _inside_and_deeper(model, backstop)
    assert backstop.limit(shallow, deep) is None
    assert backstop.limit(shallow, deep) is None
    assert len(said) == 1 and said[0].startswith("Holding:")  # Same second: once.
    clock.now = 2.0
    assert backstop.limit(shallow, deep) is None
    assert len(said) == 2


def test_backstop_lets_an_arm_inside_the_margin_work_its_way_out(rig):
    spec, model, _capsules = rig
    backstop = entry._CollisionBackstop(model, spec, emit=lambda _m: None)
    shallow, deep = _inside_and_deeper(model, backstop)
    assert backstop.limit(deep, shallow) is shallow


def test_backstop_names_the_floor(rig):
    spec, model, _capsules = rig
    said = []
    backstop = entry._CollisionBackstop(model, spec, emit=said.append)
    diving, _worst = _fold(model, backstop.capsules, avoid=False, goal=DIVE_GOAL)
    backstop.limit(_home(model).qpos.copy(), diving)
    assert said and f"/ {FLOOR} would come within" in said[0]


def test_spring_fades_in_at_influence_and_reaches_full_weight_at_margin(rig):
    _spec, _model, capsules = rig
    error, gain = capsules._spring(capsules.influence)
    assert error == 0.0 and gain == 0.0
    error, gain = capsules._spring(capsules.margin)
    assert error == pytest.approx(capsules.weight * (capsules.influence - capsules.margin))
    assert gain == pytest.approx(2 * capsules.weight)


def test_viewer_tool_runs_headless(capsys):
    _TOOL = importlib.util.spec_from_file_location(
        "view_capsule_avoidance", ROOT / "tools" / "view_capsule_avoidance.py"
    )
    tool = importlib.util.module_from_spec(_TOOL)
    _TOOL.loader.exec_module(tool)
    assert tool.main(["--hand", "none", "--frames", "120"]) == 0
    assert tool.main(["--hand", "none", "--frames", "120", "--no-avoid"]) == 0
    on, off = capsys.readouterr().out.strip().splitlines()
    assert on.startswith("avoidance ON") and off.startswith("avoidance OFF")


def test_only_the_single_arm_opts_in_to_avoidance():
    from example import teleop_sim

    for key, expected in (
        (("realman_rm75b_single", "none"), True),
        (("realman_rm75b_single", "inspire"), True),
        (("realman_rm75b", "inspire"), False),
        (("cvte", "gripper"), False),
    ):
        spec = teleop_sim._ROBOT_HAND_SPECS[key]
        model = mujoco.MjModel.from_xml_path(str(spec.scene_path))
        kwargs = teleop_sim._collision_kwargs(model, spec)
        assert bool(kwargs) is expected, key
        if kwargs:
            assert kwargs["collision"].floor_z == 0.0
