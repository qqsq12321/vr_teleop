# Quest Teleop MuJoCo

基于 Quest 3 手部追踪的机械臂遥操作系统。配合 [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) 应用使用。

## 支持的机器人配置

| 脚本 | 机器人 | 模式 | 说明 |
|------|--------|------|------|
| `teleop_kinova_gripper_sim.py` | Kinova Gen3 + Robotiq 2F-85 | 仿真 | MuJoCo 仿真，右手控制手臂+捏合控制夹爪 |
| `teleop_kinova_gripper_real.py` | Kinova Gen3 + Robotiq 2F-85 | 实物 | Kortex SDK 控制实物机械臂和夹爪 |
| `teleop_kinova_wuji_sim.py` | Kinova Gen3 + Wuji Hand | 仿真 | MuJoCo 仿真，右手控制手臂+手部重定向 |
| `teleop_kinova_wuji_real.py` | Kinova Gen3 + Wuji Hand | 实物 | Kortex SDK 控制手臂 + Wuji 灵巧手 |
| `teleop.py` | Piper 单臂 | 仿真 | 右手控制机械臂+捏合控制夹爪 |
| `teleop_bimanual.py` | Aloha 双臂 | 仿真 | 左右手分别控制两个臂 |

## 前置条件

- Quest 3 安装并运行 `hand-tracking-streamer` 应用
- Quest 与 PC 在同一局域网
- Python 3.10+，使用 conda 环境 `teleop`（`conda activate teleop`）
- MuJoCo (`pip install mujoco`)
- Wuji Hand 相关额外需要:
  - `wuji_retargeting` — 手部重定向库（来自 `wuji-retargeting` 仓库）
  - `wujihandpy` — Wuji Hand 硬件 SDK（实物控制时需要）
- Kinova 实物控制额外需要: Kinova Kortex SDK (`pip install kortex-api`)

## 运行命令

### Kinova Gen3 + Robotiq 夹爪（仿真）

```bash
cd /home/hand/teleop/quest_teleop_mujoco
PYTHONPATH=/home/hand PYTHONUNBUFFERED=1 python3 teleop_env/teleop_kinova_gripper_sim.py --port 9000
```

### Kinova Gen3 + Robotiq 夹爪（实物）

```bash
cd /home/hand/teleop/quest_teleop_mujoco
PYTHONPATH=/home/hand PYTHONUNBUFFERED=1 python3 teleop_env/teleop_kinova_gripper_real.py \
  --kinova-ip 192.168.1.10 \
  --port 9000
```

可选参数:
- `--kinova-username admin` — Kinova 登录用户名（默认 admin）
- `--kinova-password admin` — Kinova 登录密码（默认 admin）
- `--position-scale 1.0` — 手腕位移映射倍率
- `--ema-alpha 0.8` — EMA 平滑系数
- `--rot-weight 1.0` — IK 旋转权重
- `--ik-damping 0.001` — IK 阻尼系数
- `--ik-current-weight 0.1` — IK 当前姿态权重

### Kinova Gen3 + Wuji Hand（仿真）

```bash
conda activate teleop
cd /home/hand/teleop/quest_teleop_mujoco
python teleop_env/teleop_kinova_wuji_sim.py --port 9000
```

### Kinova Gen3 + Wuji Hand（实物）

```bash
conda activate teleop
cd /home/hand/teleop/quest_teleop_mujoco
python teleop_env/teleop_kinova_wuji_real.py \
  --kinova-ip 192.168.1.10 \
  --port 9000
```

可选参数:
- `--disable-arm` — 仅控制手部（不连接 Kinova 臂）
- `--disable-hand` — 仅控制机械臂（不连接 Wuji 手）
- `--hand-config path/to/config.yaml` — 指定手部重定向配置文件

### Piper 单臂（仿真）

```bash
cd /home/hand/teleop/quest_teleop_mujoco
python3 teleop_env/teleop.py --port 9000
```

### Aloha 双臂（仿真）

```bash
cd /home/hand/teleop/quest_teleop_mujoco
python3 teleop_env/teleop_bimanual.py --port 9000
```

## 工具脚本

```bash
# 将 Kinova 回到官方 Home 位姿
PYTHONPATH=/home/hand PYTHONUNBUFFERED=1 python3 move_home.py --kinova-ip 192.168.1.10

# 可视化 MuJoCo 场景（无需 Quest）
python3 teleop_env/visualize.py teleop_env/scene/scene_kinova_gen3.xml
```

## 项目结构

```
quest_teleop_mujoco/
├── teleop_env/              # 遥操作脚本
│   ├── scene/               # MuJoCo 场景文件 (XML)
│   ├── teleop_kinova_gripper_sim.py
│   ├── teleop_kinova_gripper_real.py
│   ├── teleop_kinova_wuji_sim.py
│   ├── teleop_kinova_wuji_real.py
│   ├── adaptive_analytical_quest3.yaml  # Wuji 手部重定向配置
│   ├── teleop.py
│   ├── teleop_bimanual.py
│   └── visualize.py
├── util/                    # 工具模块 (IK, 四元数, UDP 解析)
├── Kinova-kortex2_Gen3_G3L/ # Kinova Kortex SDK (vendored)
├── move_home.py             # Kinova 回 Home 工具
└── README.md
```

## Quest 端设置

- IP: PC 的局域网 IP（通过 `hostname -I` 查看）
- 端口: `9000`
- 协议: UDP
