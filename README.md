# VR Teleop

基于 VR 手部追踪的机械臂遥操作系统。支持三种输入设备：
- **Meta Quest 3** — 配合 [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) 应用，通过 UDP 传输
- **Apple Vision Pro** — 配合 [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用，通过 [avp_stream](https://github.com/Improbable-AI/VisionProTeleop) (gRPC) 传输
- **Pico 4** — 配合 `pico_relay_daemon`，通过本机 relay TCP 或直连 TCP 传输

## 支持的机器人配置

| 命令 | 机器人 | 模式 | 说明 |
|------|--------|------|------|
| `teleop_sim.py --robot kinova_gripper` | Kinova Gen3 + Robotiq 2F-85 | 仿真 | MuJoCo 仿真，右手控制手臂 + 捏合控制夹爪 |
| `teleop_real.py --robot kinova_gripper` | Kinova Gen3 + Robotiq 2F-85 | 实物 | Kortex SDK 控制实物机械臂和夹爪 |
| `teleop_sim.py --robot kinova_wuji` | Kinova Gen3 + Wuji Hand | 仿真 | MuJoCo 仿真，右手控制手臂 + 手部重定向 |
| `teleop_real.py --robot kinova_wuji` | Kinova Gen3 + Wuji Hand | 实物 | Kortex SDK 控制手臂 + Wuji 灵巧手 |
| `teleop_sim.py --robot piper` | Piper 单臂 | 仿真 | 右手控制机械臂 + 捏合控制夹爪 |
| `teleop_sim.py --robot rm65` | Realman RM65 | 仿真 | MuJoCo 仿真，右手控制 6 轴机械臂 |
| `teleop_real.py --robot rm65` | Realman RM65 | 实物 | RM API2 控制 6 轴机械臂 |
| `teleop_real.py --robot rm65_gripper` | Realman RM65 + EG2-4C2 | 实物 | RM API2 控制机械臂，右手捏合控制二指夹爪 |
| `teleop_sim.py --robot rm65_inspire` | Realman RM65 + Inspire Hand | 仿真 | MuJoCo 仿真，右手控制手臂 + Inspire 灵巧手重定向 |
| `teleop_real.py --robot rm65_inspire` | Realman RM65 + Inspire Hand | 实物 | RM API2 控制机械臂 + 串口控制 Inspire 灵巧手 |
| `teleop_sim.py --robot dexforce` | DexForce 双臂 + 双手 | 仿真 | MuJoCo 仿真，左右手控制双臂；手部配置可先留空 |
| `teleop_sim.py --robot aloha` | Aloha 双臂 | 仿真 | 左右手分别控制两个臂 |

## 前置条件

### VR 端（任选其一）

**Quest 3:**
- 安装并运行 [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) 应用
- Quest 与 PC 在同一局域网

**Apple Vision Pro:**
- 安装并运行 [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用
- Vision Pro 与 PC 在同一局域网
- PC 安装 `avp_stream>=2.50.0`（`pip install avp-stream`）

**Pico 4:**
- 启动 Pico 4 手部追踪数据发送端
- 推荐在 PC 端启动 `pico_relay_daemon`，默认监听 `127.0.0.1:63902`
- 也可使用 direct 模式，由 PC 监听 TCP `63901` 并通过 UDP `29888` 广播地址

### PC 端

- Python 3.12+（`pyproject.toml` 当前要求 `>=3.12`），conda 环境 `teleop`（`conda activate teleop`）
- MuJoCo (`pip install mujoco`)

### 第三方仓库（位于 `third_party/`）

| 仓库 | 用途 | 来源 |
|------|------|------|
| `AnyDexRetarget` | 手部关键点→灵巧手关节角重定向 | `https://gitee.com/gx_robot/AnyDexRetarget.git` (git submodule) |
| `Kinova-kortex2_Gen3_G3L` | Kinova Gen3 Kortex SDK Python 示例 | 需单独放到 `third_party/Kinova-kortex2_Gen3_G3L/` |
| `mujoco_menagerie` | MuJoCo 机器人模型（Piper, Kinova, Aloha 等） | `git@github.com:wengmister/mujoco_menagerie.git` (git submodule) |
| `RM_API2` | Realman RM65 Python SDK | 置于 `third_party/RM_API2/` |
| `rm_models` | Realman RM65 / Inspire MuJoCo 与模型资源 | 置于 `third_party/rm_models/` |

建议先初始化 submodule：

```bash
git submodule update --init --recursive
```

### 额外依赖（按功能）

- **Wuji Hand 实物控制**: `wujihandpy` — Wuji Hand 硬件 SDK
- **Kinova 实物控制**: `kortex-api` (`pip install kortex-api`)
- **Apple Vision Pro 输入**: `avp-stream>=2.50.0` (`pip install avp-stream`)
- **RM65 实物控制**: 依赖 `third_party/RM_API2/`
- **Inspire Hand 实物控制**: 串口设备，默认端口 `/dev/ttyUSB0`

## 运行命令

### Quest 3 输入（默认）

```bash
# Kinova + 夹爪（仿真）
python example/teleop_sim.py --robot kinova_gripper --port 9000

# Kinova + 夹爪（实物）
python example/teleop_real.py --robot kinova_gripper --kinova-ip 192.168.1.10 --port 9000

# Kinova + Wuji Hand（仿真）
python example/teleop_sim.py --robot kinova_wuji --port 9000

# Kinova + Wuji Hand（实物）
python example/teleop_real.py --robot kinova_wuji --kinova-ip 192.168.1.10 --port 9000

# Piper 单臂（仿真）
python example/teleop_sim.py --robot piper --port 9000

# RM65（仿真）
python example/teleop_sim.py --robot rm65 --port 9000

# RM65（实物）
python example/teleop_real.py --robot rm65 --rm65-ip 192.168.1.18 --port 9000

# RM65 + EG2-4C2（实物）
python example/teleop_real.py --robot rm65_gripper --rm65-ip 192.168.1.18 --port 9000

# RM65 + Inspire Hand（仿真）
python example/teleop_sim.py --robot rm65_inspire --port 9000

# RM65 + Inspire Hand（实物）
python example/teleop_real.py --robot rm65_inspire --rm65-ip 192.168.1.18 --inspire-port /dev/ttyUSB0 --port 9000

# Aloha 双臂（仿真）
python example/teleop_sim.py --robot aloha --port 9000
```

### Apple Vision Pro 输入

在任意命令后追加 `--input-source avp --avp-ip <Vision Pro IP>`:

```bash
# Kinova + 夹爪（仿真）
python example/teleop_sim.py --robot kinova_gripper --input-source avp --avp-ip 192.168.5.32

# Kinova + 夹爪（实物）
python example/teleop_real.py --robot kinova_gripper --input-source avp --avp-ip 192.168.5.32 --kinova-ip 192.168.1.10

# Kinova + Wuji Hand（仿真）
python example/teleop_sim.py --robot kinova_wuji --input-source avp --avp-ip 192.168.5.32

# Kinova + Wuji Hand（实物）
python example/teleop_real.py --robot kinova_wuji --input-source avp --avp-ip 192.168.5.32 --kinova-ip 192.168.1.10

# RM65（实物）
python example/teleop_real.py --robot rm65 --input-source avp --avp-ip 192.168.5.32 --rm65-ip 192.168.1.18

# RM65 + EG2-4C2（实物）
python example/teleop_real.py --robot rm65_gripper --input-source avp --avp-ip 192.168.5.32 --rm65-ip 192.168.1.18

# RM65 + Inspire Hand（仿真）
python example/teleop_sim.py --robot rm65_inspire --input-source avp --avp-ip 192.168.5.32

# RM65 + Inspire Hand（实物）
python example/teleop_real.py --robot rm65_inspire --input-source avp --avp-ip 192.168.5.32 --rm65-ip 192.168.1.18 --inspire-port /dev/ttyUSB0

# Aloha 双臂（仿真）
python example/teleop_sim.py --robot aloha --input-source avp --avp-ip 192.168.5.32
```

### Pico 4 输入

relay 模式默认连接 `127.0.0.1:63902`：

```bash
# Kinova + 夹爪（仿真）
python example/teleop_sim.py --robot kinova_gripper --input-source pico4

# Kinova + Wuji Hand（仿真）
python example/teleop_sim.py --robot kinova_wuji --input-source pico4

# Kinova + Wuji Hand（仿真，vector 优化器）
python example/teleop_sim.py --robot kinova_wuji --input-source pico4 --hand-config config/pico4/vector/pico4_wuji_hand.yaml

# RM65 + Inspire Hand（仿真）
python example/teleop_sim.py --robot rm65_inspire --input-source pico4

# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --robot dexforce --input-source pico4

# Aloha 双臂（仿真）
python example/teleop_sim.py --robot aloha --input-source pico4
```

direct 模式：

```bash
python example/teleop_sim.py --robot kinova_gripper --input-source pico4 --pico4-mode direct --pico4-port 63901
```

停止方式说明：

- Apple Vision Pro 实物路径当前支持左手握拳保持约 3 秒停止
- 其余模式默认使用 `Ctrl+C`

## 可选参数

输入源:
- `--input-source quest3|avp|pico4` — 输入设备（默认 quest3）
- `--avp-ip <IP>` — Apple Vision Pro IP 地址（仅 avp 模式）
- `--pico4-mode relay|direct` — Pico 4 输入模式（默认 relay）
- `--pico4-relay-host 127.0.0.1` — Pico 4 relay 主机
- `--pico4-relay-port 63902` — Pico 4 relay 端口
- `--pico4-port 63901` — Pico 4 direct 模式 TCP 端口
- `--pico4-broadcast-port 29888` — Pico 4 direct 模式 UDP 广播端口

通用:
- `--port 9000` — Quest 3 UDP 端口
- `--position-scale` — 手腕位移映射倍率（各机器人有不同默认值）
- `--ema-alpha 0.8` — EMA 平滑系数
- `--rot-weight 1.0` — IK 旋转权重
- `--ik-damping 0.001` — IK 阻尼系数
- `--ik-current-weight 0.1` — IK 当前姿态权重

仿真专用:
- `--scene path/to/scene.xml` — 覆盖默认场景
- `--site site_name` — 覆盖末端执行器 site 名称
- `--hand-config path/to/config.yaml` — 指定手部重定向配置文件（`kinova_wuji` / `rm65_inspire` / `dexforce` 可选）
- `--hand-side left|right` — 手部侧向（灵巧手重定向场景，默认 `right`）

实物专用:
- `--kinova-ip 192.168.1.10` — Kinova IP
- `--kinova-username admin` — Kinova 用户名
- `--kinova-password admin` — Kinova 密码
- `--rm65-ip 192.168.1.18` — RM65 IP
- `--hand-config path/to/config.yaml` — 指定 Wuji 手重定向配置文件（当前主要用于 `kinova_wuji`；`rm65_inspire` 实物仍使用内置默认配置；`dexforce` 可先留空）
- `--disable-arm` — 仅控制手部（`kinova_wuji` 或 `rm65_inspire`）
- `--disable-hand` — 仅控制机械臂（当前主要用于 `kinova_wuji`）
- `--inspire-port /dev/ttyUSB0` — Inspire 手串口设备
- `--inspire-baudrate 115200` — Inspire 手串口波特率
- `--inspire-hand-id 1` — Inspire 手串口协议手 ID

## 工具脚本

```bash
# 将 Kinova 回到官方 Home 位姿
PYTHONPATH=/home/hand PYTHONUNBUFFERED=1 python3 util/arm_move_home.py --kinova-ip 192.168.1.10

# 可视化 MuJoCo 场景（无需 Quest）
python3 viz/visualize.py example/scene/scene_kinova_gen3.xml
```

## 项目结构

```
vr_teleop/
├── config/                      # 本项目自有配置
│   └── pico4/                      # Pico 4 手部重定向配置
├── example/                     # 遥操作入口脚本
│   ├── teleop_sim.py               # 仿真遥操作（MuJoCo viewer）
│   ├── teleop_real.py              # 实物遥操作（Kortex SDK）
│   └── scene/                      # MuJoCo 场景文件 (XML)
├── util/                        # 核心模块
│   ├── ik.py                       # 逆运动学求解器
│   ├── quaternion.py               # 四元数运算 + VR→机器人坐标变换
│   ├── udp_socket.py               # UDP 收发 + Quest 数据包解析
│   ├── avp_input.py                # Apple Vision Pro 输入适配器（avp_stream 封装）
│   ├── pico4_input.py              # Pico 4 输入适配器（relay/direct TCP）
│   ├── wrist_tracker.py            # 腕部残差跟踪（EMA 平滑 + deadband）
│   ├── hand_retarget.py            # 灵巧手重定向（landmarks → 关节角）
│   └── arm_move_home.py            # Kinova 回 Home 工具脚本
├── viz/                         # 可视化
│   └── visualize.py                # MuJoCo 场景查看器
├── third_party/                 # 第三方依赖
│   ├── AnyDexRetarget/             # 手部重定向库
│   ├── Kinova-kortex2_Gen3_G3L/    # Kinova Kortex SDK
│   ├── mujoco_menagerie/           # MuJoCo 机器人模型
│   ├── RM_API2/                    # Realman RM65 SDK
│   └── rm_models/                  # RM65 / Inspire 模型资源
└── README.md
```

## VR 端设置

### Quest 3
- IP: PC 的局域网 IP（通过 `hostname -I` 查看）
- 端口: `9000`
- 协议: UDP

### Apple Vision Pro
- 打开 Tracking Streamer 应用，点 Start
- 记下 Vision Pro 的 IP 地址（设置 → Wi-Fi → 已连接网络）
- PC 端使用 `--input-source avp --avp-ip <IP>` 连接

### Pico 4
- relay 模式：确认 `pico_relay_daemon` 正在运行，PC 端使用 `--input-source pico4`
- direct 模式：PC 端使用 `--input-source pico4 --pico4-mode direct`，Pico 4 端连接 PC 广播出来的地址
