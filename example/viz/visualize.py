from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import mujoco
from mujoco import viewer

from utils.robots.galbot import GalbotInspireSpec


_SCENE_ALIASES = {
    "dexforce": "example/scene_config/scene_dexforce.xml",
    "galbot": "example/scene_config/scene_galbot_inspire.xml",
    "galbot_inspire": "example/scene_config/scene_galbot_inspire.xml",
}


def _write_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...], values) -> None:
    for name, value in zip(names, values):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint '{name}' not found.")
        data.qpos[model.jnt_qposadr[jid]] = float(value)


def _apply_visual_home(model: mujoco.MjModel, data: mujoco.MjData, scene_name: str, xml_path: Path) -> None:
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)

    galbot_spec = GalbotInspireSpec()
    galbot_scene_paths = {
        galbot_spec.scene_path.resolve(),
        (ROOT_DIR / "example" / "scene_config" / "scene_galbot_inspire.xml").resolve(),
        (ROOT_DIR / "example" / "scene_config" / "galbot_inspire_visual.urdf").resolve(),
    }
    if scene_name in {"galbot", "galbot_inspire"} or xml_path in galbot_scene_paths:
        _write_joint_qpos(model, data, galbot_spec.torso_joint_names, galbot_spec.torso_home_qpos)
        _write_joint_qpos(model, data, galbot_spec.head_joint_names, galbot_spec.head_home_qpos)
        _write_joint_qpos(model, data, galbot_spec.arm_joint_names("left"), galbot_spec.arm_home_qpos("left"))
        _write_joint_qpos(model, data, galbot_spec.arm_joint_names("right"), galbot_spec.arm_home_qpos("right"))

    if model.nu:
        data.ctrl[: min(model.nu, model.nq)] = data.qpos[: min(model.nu, model.nq)]
    mujoco.mj_forward(model, data)


def _configure_camera(vis, model: mujoco.MjModel, scene_name: str) -> None:
    if scene_name in {"galbot", "galbot_inspire"}:
        vis.cam.azimuth = 135
        vis.cam.elevation = -18
        vis.cam.distance = 2.8
        vis.cam.lookat[:] = (0.28, 0.0, 0.95)
        return

    vis.cam.azimuth = model.vis.global_.azimuth
    vis.cam.elevation = model.vis.global_.elevation
    vis.cam.distance = model.stat.extent * 1.5
    vis.cam.lookat[:] = model.stat.center


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo scene.")
    parser.add_argument(
        "scene",
        nargs="?",
        default="dexforce",
        help=(
            "Path to a MuJoCo XML scene file, or a short alias: "
            + ", ".join(_SCENE_ALIASES)
        ),
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Advance physics while visualizing. This is the default unless --static is set.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Do not advance physics. Useful for fixed model inspection.",
    )
    args = parser.parse_args()
    scene_name = args.scene
    args.scene = _SCENE_ALIASES.get(args.scene, args.scene)

    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _apply_visual_home(model, data, scene_name, xml_path)

    with viewer.launch_passive(model, data) as vis:
        _configure_camera(vis, model, scene_name)
        should_step = not args.static
        while vis.is_running():
            step_start = time.time()
            if should_step:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            vis.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
