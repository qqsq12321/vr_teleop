# CVTE CARM D3 assets

This directory holds the CVTE "CARM D3" flexible dual-arm body used by
`scene_cvte_carm_d3.xml`. The arm is a fixed-base 14-DOF dual-arm (7 revolute
joints per side) ending in a single-DOF parallel gripper per side; there is no
torso, head, or mobile base.

## Source and generated files

- `source/urdf/carm_d3.urdf`: the vendor URDF (from CVTE's `carm_d3` ROS
  package), lightly touched:
  - `package://carm_d3/meshes/...` rewritten to the relative `../../meshes/...`
    path so the file has no dependency on a ROS package path.
  - `<mujoco><compiler fusestatic="false"/></mujoco>` injected so MuJoCo keeps
    the vendor's fixed-joint links (e.g. `gripper_base-{LEFT,RIGHT}`) as named
    bodies instead of fusing them into their parents.
  - No other structural edits: link topology, inertials, and joint limits are
    vendor-original.
- `meshes/*.STL`: vendor meshes, copied unchanged. They are imported
  **visual-only** (not reused as collision geometry): the vendor URDF points
  both `<visual>` and `<collision>` at the same dense STL (50k-170k faces per
  link), and this project's teleop scenes already disable all robot
  self/environment collision (see `scene_dexforce.xml`'s `contype="0"
  conaffinity="0"` classes), so importing a duplicate collision mesh set would
  only double load time for no benefit.
- The vendor URDF's `<mimic joint="gripper_right_joint-{SIDE}" multiplier="1"
  offset="0"/>` tag on `gripper_left_joint-{SIDE}` is **not** converted to a
  MuJoCo equality constraint by the URDF importer (verified: `neq == 0` after
  compiling the raw URDF). `scene_cvte_carm_d3.xml` adds the equivalent
  `<equality><joint .../></equality>` by hand during the scene build step.

Rebuild the scene from these checked-in sources:

```bash
python tools/build_cvte_carm_d3_scene.py
```

The generated scene uses project-relative paths and has no runtime dependency
on the original ROS package path.
