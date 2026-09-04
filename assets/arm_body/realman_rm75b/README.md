# RealMan RM75-B assets

This directory contains the RealMan RM75-B mobile dual-arm body used by
`scene_realman_rm75b_inspire.xml`. The robot combination is intentionally fixed
to the project's Inspire RH56DFX left/right hands.

## Source and generated files

- `source/urdf/`: xacro sources imported from the supplied
  `dual_75B_arm_robot` ROS package.
- `meshes/visual/`: runtime visual meshes used by MuJoCo.
- `body_base_link.STL` is simplified on a deterministic 1 mm vertex grid so it
  stays below MuJoCo's STL face limit. The other meshes are copied unchanged.
- The vendor's fixed `l_hand_base_link.STL` and `r_hand_base_link.STL` are not
  imported because the scene mounts the articulated Inspire model instead.

Re-import and rebuild from a newer vendor package:

```bash
python tools/build_realman_rm75b_assets.py \
  --vendor-dir /path/to/dual_75B_arm_robot
```

After the first import, the scene can be rebuilt from checked-in sources/assets:

```bash
python tools/build_realman_rm75b_assets.py
```

The generated scene uses project-relative paths and has no runtime dependency
on the original ROS package path. Vendor wheel joints are fixed for this
stationary teleoperation scene; vendor link topology and arm mounting poses are
otherwise preserved.
