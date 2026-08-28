# VR Teleop

基于 VR 手部追踪的机械臂遥操作系统。支持三种输入设备：
- **Meta Quest 3** — 配合 [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) 应用，通过 UDP 传输
- **Apple Vision Pro** — 配合 [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用，通过 [avp_stream](https://github.com/Improbable-AI/VisionProTeleop) (gRPC) 传输
- **Pico 4** — 配合 `pico_relay_daemon`，通过本机 relay TCP 或直连 TCP 传输

## 支持的机器人配置

| 命令 | 机器人 | 模式 | 说明 |
|------|--------|------|------|
| `teleop_sim.py` | DexForce 双臂 + 双手 | 仿真 | MuJoCo XML 场景，左右手控制双臂；手部重定向为 linker_l20 |
| `teleop_real.py` | DexForce 双臂 + 双手 | 实物 | 骨架已就绪，需适配 DexForce 真机 SDK |

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
| `AnyDexRetarget` | 手部关键点→灵巧手关节角重定向（linker_l20） | `https://gitee.com/gx_robot/AnyDexRetarget.git` (git submodule) |

建议先初始化 submodule：

```bash
git submodule update --init --recursive
```

### 额外依赖（按功能）

- **Apple Vision Pro 输入**: `avp-stream>=2.50.0` (`pip install avp-stream`)
- **DexForce 实物控制**: 待适配（详见 `example/teleop_real.py` 中 `DexForceRealArm` TODO）

## 运行命令

### Pico 4 输入（本文档默认）

本文档优先介绍 Pico 4；程序仍保留原始 CLI 默认值 `quest3`，因此使用 Pico 4 时需要显式指定输入源。

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source pico4 --pico4-mode direct
```

默认由 PC 监听 TCP `63901`，并通过 UDP `29888` 广播 USB/Wi-Fi 地址。

### Quest 3 输入

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source quest3 --port 9000

# DexForce 双臂 + 双手（实物骨架）
python example/teleop_real.py --dexforce-ip 192.168.1.50 --port 9000
```

### Apple Vision Pro 输入

在任意命令后追加 `--input-source avp --avp-ip <Vision Pro IP>`:

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source avp --avp-ip 192.168.5.32

# DexForce 双臂 + 双手（实物骨架）
python example/teleop_real.py --input-source avp --avp-ip 192.168.5.32 --dexforce-ip 192.168.1.50
```

### Pico 4 relay 模式

项目提供独立的 `example/pico4_daemon.py`。它负责保持 PICO 连接，并将数据转发到 `127.0.0.1:63902`，因此启动或重启仿真时无需让 PICO 重新连接。

终端 1：

```bash
python example/pico4_daemon.py
```

终端 2：

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source pico4 --pico4-mode relay

# DexForce 双臂 + 双手（实物骨架）
python example/teleop_real.py --input-source pico4 --dexforce-ip 192.168.1.50
```

停止 daemon：在终端 1 按 `Ctrl+C`。

停止方式说明：

- Apple Vision Pro 实物路径当前支持左手握拳保持约 3 秒停止
- 其余模式默认使用 `Ctrl+C`

## 可选参数

输入源:
- `--input-source quest3|avp|pico4` — 输入设备（CLI 默认 quest3；本文档默认介绍 pico4）
- `--avp-ip <IP>` — Apple Vision Pro IP 地址（仅 avp 模式）
- `--pico4-mode relay|direct` — Pico 4 输入模式（CLI 默认 relay；本文档推荐 direct）
- `--pico4-relay-host 127.0.0.1` — Pico 4 relay 主机
- `--pico4-relay-port 63902` — Pico 4 relay 端口
- `--pico4-port 63901` — Pico 4 direct 模式 TCP 端口
- `--pico4-broadcast-port 29888` — Pico 4 direct 模式 UDP 广播端口

通用:
- `--port 9000` — Quest 3 UDP 端口
- `--position-scale` — 手腕位移映射倍率（默认 3.0）
- `--ema-alpha 0.8` — EMA 平滑系数
- `--rot-weight 1.0` — IK 旋转权重
- `--ik-damping 0.001` — IK 阻尼系数
- `--ik-current-weight 0.1` — IK 当前姿态权重

仿真专用:
- `--scene path/to/scene.xml` — 覆盖默认场景（默认 `example/scene_config/scene_dexforce.xml`）
- `--hand-config path/to/config.yaml` — 指定 linker_l20 手部重定向配置文件

实物专用:
- `--dexforce-ip 192.168.1.50` — DexForce 机器人 IP
- `--hand-config path/to/config.yaml` — 指定 linker_l20 手部重定向配置文件

## 工具脚本

```bash
# 可视化 MuJoCo 场景（无需 VR）
python3 example/viz/visualize.py dexforce
# 或直接指定 XML
python3 example/viz/visualize.py example/scene_config/scene_dexforce.xml
```

## 项目结构

```
vr_teleop/
├── utils/                       # 核心算法、适配器、机器人规格
│   ├── core/
│   │   ├── types.py               # 共享数据结构
│   │   ├── quaternion.py          # 四元数运算 + 坐标变换
│   │   ├── body/
│   │   │   ├── ik.py              # 机械臂/本体 IK
│   │   │   └── wrist_tracker.py   # 腕部残差跟踪
│   │   └── hand/
│   │       └── retarget.py        # 灵巧手重定向
│   ├── adapters/
│   │   ├── input/
│   │   │   ├── quest3.py
│   │   │   ├── avp.py
│   │   │   └── pico4.py
│   │   └── output/
│   │       ├── mujoco_sim.py
│   │       └── dexforce_real.py
│   └── robots/
│       └── dexforce.py            # DexForce 关节映射、home pose、配置路径
├── assets/                      # 机器人/物体模型资产
│   ├── arm_body/dexforce/          # DexForce 本体 mesh、贴图、URDF
│   ├── dex_hand/linker_l20/        # linker_l20 手部 URDF/MJCF、visual/collision mesh
│   └── objects/                    # 示例物体 mesh 和预览图
├── example/                     # 遥操作入口和示例配置
│   ├── teleop_sim.py
│   ├── teleop_real.py
│   ├── scene_config/               # MuJoCo XML 场景配置
│   └── viz/visualize.py            # MuJoCo 场景查看器
├── third_party/                 # 第三方依赖
│   └── AnyDexRetarget/             # 手部重定向库 (git submodule)
└── README.md
```

## 架构总览

系统采用分层设计，从 VR 输入到机器人执行形成统一管线：

```
┌──────────────────────────────────────────────────────────┐
│  入口层  example/teleop_sim.py · teleop_real.py          │
│  DexForce 双臂 + 双手 (linker_l20) 路径                  │
├──────────────────────────────────────────────────────────┤
│  输入适配层                                              │
│  utils/adapters/input/quest3.py   Quest 3 (UDP)          │
│  utils/adapters/input/avp.py      Vision Pro (gRPC)      │
│  utils/adapters/input/pico4.py    Pico 4 (TCP)           │
│  → 统一输出: wrist_pose / landmarks_mediapipe              │
├──────────────────────────────────────────────────────────┤
│  坐标变换层  utils/core/quaternion.py                     │
│  VR 坐标系 → 机器人坐标系；四元数运算                    │
├──────────────────────────────────────────────────────────┤
│  腕部跟踪层  utils/core/body/wrist_tracker.py  (WristTracker) │
│  残差跟踪 + EMA 平滑 + deadband + base_xmat 变换         │
├──────────────────────────────────────────────────────────┤
│  求解层                                                  │
│  utils/core/body/ik.py        Levenberg-Marquardt IK     │
│  utils/core/hand/retarget.py  灵巧手重定向 (AnyDexRetarget) │
├──────────────────────────────────────────────────────────┤
│  执行层                                                  │
│  仿真: MuJoCo viewer + data.qpos                         │
│  实物: DexForceRealArm (TODO: 适配真机 SDK)              │
├──────────────────────────────────────────────────────────┤
│  资源层                                                  │
│  example/scene_config/scene_dexforce.xml   MuJoCo 场景           │
│  assets/arm_body + assets/dex_hand + assets/objects  资产 │
│  third_party/AnyDexRetarget/example/config/adaptive  重定向配置 │
└──────────────────────────────────────────────────────────┘
```

### 关键设计

- **Sim/Real 双入口**：
  - `teleop_sim.py` — `viewer.launch_passive` 可视化，`DexForceArmController.apply_qpos` 直接写 MuJoCo `data.qpos`，含 Pico4 头追→躯干/颈部联动
  - `teleop_real.py` — `DexForceArmController.step` 走笛卡尔平滑层（`cmd_pos += (target-cmd_pos)*gain` + SLERP）后 IK，再由 `DexForceRealArm` 发送真机指令；`DexForceRealArm` 为待适配骨架
- **WristTracker 共享**：仿真与实物共用同一残差跟踪器，首帧锁定初始位姿，后续残差经 EMA 平滑后乘 `position_scale` 累加到目标位姿。实物路径额外加笛卡尔平滑层与 deadband。
- **手部重定向**：`HandRetargeter` 封装 `AnyDexRetarget`（git submodule），按 input_source 路由到 linker_l20 YAML 配置。Quest3 用 63 维原始 landmarks（经 `landmarks_to_mediapipe` 转 RH），AVP/Pico4 用 21×3 mediapipe 格式。

### 数据流（以 Quest3 + DexForce 仿真为例）

```
Quest3 UDP 包
  → utils.adapters.input.quest3.parse_left/right_wrist_pose / parse_left/right_landmarks
  → quaternion.transform_vr_to_robot_pose (腕部位姿)
  → WristTracker.update (残差 + EMA → target_pos / target_quat)
  → ik.solve_body_pose_ik (LM 迭代 → q_sol)
  → DexForceArmController.apply_qpos → data.qpos[arm]
  → utils.core.hand.retarget.HandRetargeter.retarget(landmarks) → 21 维手部关节
  → apply_qpos 映射 → data.qpos[hand]
  → mujoco.mj_forward + viewer.sync
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
