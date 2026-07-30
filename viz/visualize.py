from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
from mujoco import viewer


_SCENE_ALIASES = {
    "piper": "example/scene/scene_piper.xml",
    "kinova_gripper": "example/scene/scene_kinova_gen3.xml",
    "kinova_wuji": "example/scene/scene_kinova_gen3_wuji.xml",
    "rm65": "example/scene/scene_rm65.xml",
    "rm65_inspire": "example/scene/scene_rm65_inspire.xml",
    "rm65_inspire_dual": "example/scene/scene_rm65_inspire_dual.xml",
    "dexforce": "example/scene/scene_dexforce.xml",
    "aloha": "example/scene/aloha/scene.xml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo scene.")
    parser.add_argument(
        "scene",
        nargs="?",
        default="piper",
        help=(
            "Path to a MuJoCo XML scene file, or a short alias: "
            + ", ".join(_SCENE_ALIASES)
        ),
    )
    args = parser.parse_args()
    args.scene = _SCENE_ALIASES.get(args.scene, args.scene)

    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        data.ctrl[:model.nu] = data.qpos[:model.nu]
    else:
        desired_qpos = [0.0, 0.9, -0.9, 0.0, 0.4, 0.0, 0.0]
        if model.nq >= len(desired_qpos):
            data.qpos[: len(desired_qpos)] = desired_qpos
        if model.nu >= len(desired_qpos):
            data.ctrl[: len(desired_qpos)] = desired_qpos
    mujoco.mj_forward(model, data)

    with viewer.launch_passive(model, data) as vis:
        vis.cam.azimuth = model.vis.global_.azimuth
        vis.cam.elevation = model.vis.global_.elevation
        vis.cam.distance = model.stat.extent * 1.5
        vis.cam.lookat[:] = model.stat.center
        while vis.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            vis.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
