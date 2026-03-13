from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
from mujoco import viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo scene.")
    parser.add_argument(
        "scene",
        nargs="?",
        default=str("teleop_env/scene/scene_piper.xml"),
        help="Path to a MuJoCo XML scene file.",
    )
    args = parser.parse_args()

    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    else:
        desired_qpos = [0.0, 0.9, -0.9, 0.0, 0.4, 0.0, 0.0]
        if model.nq >= len(desired_qpos):
            data.qpos[: len(desired_qpos)] = desired_qpos
        if model.nu >= len(desired_qpos):
            data.ctrl[: len(desired_qpos)] = desired_qpos
    mujoco.mj_forward(model, data)

    with viewer.launch_passive(model, data) as vis:
        while vis.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            vis.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
