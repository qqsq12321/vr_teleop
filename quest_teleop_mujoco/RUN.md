# Quest Teleop MuJoCo 运行指南

## 前置条件

- Quest 端安装并运行 `hand-tracking-streamer` 应用
- Quest 与 PC 在同一局域网
- PC 当前 IP: `192.168.5.30`（可通过 `hostname -I` 查看）

## 单臂遥操作（Piper）

```bash
conda activate teleop
cd /home/hand/teleop/quest_teleop_mujoco
python3 teleop_env/teleop.py --port 9000
```

- 机器人: Piper 单臂
- 协议: UDP
- 控制: 右手控制机械臂，拇指食指捏合控制夹爪

## 单臂遥操作（Kinova Gen3）

```bash
conda activate teleop
cd /home/hand/teleop/quest_teleop_mujoco
python3 teleop_env/teleop_kinova_gripper_sim.py --port 9000
```

- 机器人: Kinova Gen3 7-DOF + Robotiq 2F-85 夹爪
- 协议: UDP
- 控制: 右手控制机械臂，拇指食指捏合控制夹爪（捏合=闭合，张开=松开）

验证场景加载（无需 Quest 连接）：

```bash
python3 teleop_env/visualize.py teleop_env/scene/scene_kinova_gen3.xml
```

## 双臂遥操作（Aloha）

```bash
conda activate teleop
cd /home/hand/teleop/quest_teleop_mujoco
python3 teleop_env/teleop_bimanual.py --port 9000
```

- 机器人: Aloha 双臂
- 协议: UDP
- 控制: 左手控制左臂，右手控制右臂

## Quest 端设置

- IP: `192.168.5.30`
- 端口: `9000`
- 协议: UDP


