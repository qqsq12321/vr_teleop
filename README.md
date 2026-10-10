# VR Teleop

基于 VR 手部追踪的机械臂遥操作系统。支持三种输入设备：
- **Meta Quest 3** — 配合 [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) 应用，通过 UDP 传输
- **Apple Vision Pro** — 配合 [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用，通过 [avp_stream](https://github.com/Improbable-AI/VisionProTeleop) (gRPC) 传输
- **Pico 4** — 配合 `pico_relay_daemon`，通过本机 relay TCP 或直连 TCP 传输

## 支持的机器人配置

| 命令 | 机器人 | 模式 | 说明 |
|------|--------|------|------|
| `teleop_sim.py` | DexForce 双臂 + 双手 | 仿真 | MuJoCo XML 场景，左右手控制双臂；手部重定向为 linker_l20 |
| `teleop_sim.py --robot realman_rm75b` | RealMan RM75-B + Inspire RH56DFX | 仿真 | `--hand` 默认 inspire |
| `teleop_sim.py --robot realman_rm75b --hand linker_l20` | RealMan RM75-B + Linker L20 | 仿真 | 仅仿真，无真机后端 |
| `teleop_real.py` | RealMan RM75-B + Inspire RH56DFX | 实物 | 按 master 的 RM65 流程适配双臂七关节，先到初始姿态，再 Pico → IK → CANFD；无口令确认 |
| `teleop_realman_real.py` | RealMan RM75-B + Inspire RH56DFX | 兼容调试入口 | 独立的旧检查/离线预览流程；日常真机入口不调用它 |

## RealMan + Inspire 真机入口

日常真机遥操使用这个入口（**连接真机后先运动到初始姿态，不等待 Pico 追踪，无口令确认**）：

```bash
conda activate vr_teleop

# 手柄遥操 + 夹爪（推荐）
python example/teleop_real.py --controller --ee gripper

# 手柄遥操，只控制双臂
python example/teleop_real.py --controller

# 手势追踪遥操（原有模式，用 Pico 手部追踪而非手柄）
python example/teleop_real.py
```

默认双臂；`--arm left|right` 用于可选的单臂控制。
运行前确认硬件状态、关节零位/方向、工作区及急停。

### 手柄键位（`--controller`）

左右手柄分别控制同侧机械臂与夹爪：

| 按键 | 功能 |
|------|------|
| grip（侧面握把键） | **死手开关**：按住才下发机械臂运动；松开立即停止并清除位姿基准 |
| trigger（食指扳机） | 夹爪控制（需 `--ee gripper`）：按住闭合，松开张开 |
| menu | 退出程序，等同 Ctrl+C |

松开 grip 后重新按住时，会以当前手柄位置作为新的零点重新对齐，不会跳变。
方向映射使用 Pico 固定坐标系，**换一个朝向重新抓取不会重定向控制轴**。

`--ee gripper` 走 RealMan 官方两指夹爪接口 `Set_Gripper_Position`（开口度 `1..1000`），
启动时用 `Set_Gripper_Release(speed=1000)` 设为最大速度；这条路径不使用
Inspire 灵巧手的 `Set_Hand_Angle`。`--ee gripper` 必须与 `--controller` 同时使用。
控制台已抑制厂商 SDK 的逐条 INFO 日志，仅保留其警告和错误。

### 控制参数

`example/teleop_real.py` 直接按本地 `master` 的 RM65 控制流程实现：
- 腕跟踪 EMA `0.5`、位移倍率 `1.5`；笛卡尔位置/旋转增益均为 `0.08`。
- `solve_pose_ik` 的 `rot_weight=1.0`、`damping=0.001`、`current_q_weight=0.3`；
  用上一次 IK 结果作为下一次初值，取每臂七关节，转角度后 `Movej_CANFD(..., follow=True)`。
- 名义周期 `20ms`，只补足剩余休眠；不再另加关节低速限制、固定 `100ms` 退出或 `150ms` 新帧退出。
- 与主分支一致：超过 `250ms` 没有有效 getter 结果时跳过该臂下发，不退出；输入恢复后沿用原基准。
  **Pico getter 会返回缓存，因此这不是网络断流检测。** 已有手指目标仍每轮下发，不受臂的这一超时分支控制。

启动时先调用 `Movej_Cmd(q_deg, v=20, r=0, trajectory_connect=0, block=True)`，
双臂依次到达项目 RM75 的初始姿态，再重新读取实际关节角建立 FK/IK 和 Pico 基准。
初始七关节目标来自 `utils/robots/realman_rm75b.py` 的 `LEFT_ARM_HOME_QPOS` / `RIGHT_ARM_HOME_QPOS`
（弧度，取自现场实测姿态，右臂为左臂逐关节镜像并单独调整 J7），不是 RM65 的六轴角度；
不执行编码器校零或自动上电。
`v=20` 仅是主分支启动 MoveJ 的速度百分比，不改变后续 CANFD 跟随参数。
初始化失败不会进入遥操或下发手指指令。
Ctrl+C / SDK 报错时尽力停两臂并断开；不自动取消手指目标，也不保证 SDK 阻塞或进程崩溃时能停机。
**当前初始姿态在 MuJoCo 模型的自碰撞检查中会报双手间隙不足；日常入口不做该检查，
使用前请自行确认两臂到位后的实际间隙。**

网络、Pico、控制参数在 `config/realman_real.yaml`。日常入口不调用旧调试循环、
`SafetySupervisor` 或模型碰撞检查；YAML 的 `safety` 与 `hands.rate_hz` 仅供旧调试入口使用。
仿真入口不改动；真机只使用 MuJoCo 做 FK/IK，不启动仿真或查看器。
详细说明见 [RealMan 真机使用说明](docs/realman_real.md)。

## RealMan + Linker L20 仿真组合

**仅仿真，没有对应真机后端。** 真机路径只支持 Inspire（见上文 `teleop_real.py`）。

```bash
conda activate vr_teleop
python example/teleop_sim.py --robot realman_rm75b --hand linker_l20 \
    --input-source avp --avp-ip <Vision Pro IP>
```

场景 `example/scene_config/scene_realman_rm75b_l20.xml` 由
`tools/build_realman_rm75b_l20_scene.py` 生成：把 `assets/dex_hand/linker_l20/`
下的 L20 手接到 `scene_realman_rm75b_inspire.xml` 的机械臂上，替换 Inspire。
两份输入场景都已入库，所以重建**不需要** xacro 或厂商 ROS 包
（这点和重建臂本体的 `build_realman_rm75b_assets.py` 不同）：

```bash
python tools/build_realman_rm75b_l20_scene.py --check
```

`--check` 校验两处：手部坐标系是否符合臂场景既有的 Inspire 约定
（手指 -Y、食指→小指 +Z/-Z），以及初始姿态下有无自碰撞。

与 Inspire 组合的差异：

- **手部自由度 21 对 12。** L20 每只手 21 个关节，均带 position 执行器；
  整个模型 `nq=nv=nu=58`（Inspire 版是 40）。重定向映射表与
  `utils/robots/dexforce.py` 完全一致，因为两个场景内嵌的是同一份 L20 模型。
- **`r_joint1` 减了 2π**（`3.96542297` → `-2.31776234` rad）。FK 逐位相同，
  但原值超出模型 ±3.11 限位，会被 `--control-mode ctrl` 截断到 3.11，
  导致右臂起始位姿和 `qpos` 模式不一致。
- **`l_joint7`（腕滚转）改为 +75°**，使左手掌心朝下，与右手一致。
  Inspire spec 的 -88.6° 会让左掌朝上——在 Inspire 场景里同样可复现，
  不是 L20 带来的；该场景入库的 keyframe 早于那次 home 改动，掩盖了这一点。
  J7 轴过 TCP site，所以只转掌心朝向，末端位置分毫不动。
- **两手间距更小。** L20 比 Inspire 每侧长约 48mm，初始姿态下中指尖间距
  0.172m（Inspire 为 0.249m）。实测无接触（`ncon=0`），但余量比 Inspire 小。
- **手部几何是纯视觉的**（`contype=0 conaffinity=0`），和臂场景里的 Inspire 一致，
  因此场景不会检测手指自碰撞。

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

- Python 3.12+（`pyproject.toml` 当前要求 `>=3.12`），conda 环境 **`vr_teleop`**（`conda activate vr_teleop`）
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
- **DexForce 实物后端**: `utils/adapters/output/dexforce_real.py` 仍是未接 SDK 的占位实现；`teleop_real.py` 已改用于 RealMan。

## 运行命令

### Pico 4 输入（本文档默认）

仿真入口默认输入仍为 `quest3`，使用 Pico 4 时需显式指定；RealMan 真机入口固定使用 Pico，模式由 YAML 配置。

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source pico4 --pico4-mode direct
```

默认由 PC 监听 TCP `63901`，并通过 UDP `29888` 广播 USB/Wi-Fi 地址。

### Quest 3 输入

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source quest3 --port 9000

```

### Apple Vision Pro 输入

在仿真命令后追加 `--input-source avp --avp-ip <Vision Pro IP>`（RealMan 精简真机入口仅接 Pico）:

```bash
# DexForce 双臂 + 双手（仿真）
python example/teleop_sim.py --input-source avp --avp-ip 192.168.5.32

# RealMan RM75-B + Linker L20（仿真）
python example/teleop_sim.py --robot realman_rm75b --hand linker_l20 \
    --input-source avp --avp-ip 192.168.5.32
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

# RealMan 真机（会连接硬件并先运动到初始姿态；不等待 Pico，无口令确认，不是仿真）
python example/teleop_real.py --controller --ee gripper
```

停止 daemon：在终端 1 按 `Ctrl+C`。

停止方式说明：

- RealMan 真机使用 `Ctrl+C` 或手柄 menu 键；SDK 报错会退出并尽力停止两臂。无有效 getter 结果超过 250ms 仅跳过臂下发，恢复后继续；缓存及手指目标行为见上文。
- 手柄模式下松开 grip 会立即停止下发该臂运动，但这是软件死手开关，**不能替代实体急停**。
- 其余入口的停止手势依各自实现；不要依赖手势替代实体急停。

## 仿真入口可选参数

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
- `--robot dexforce|galbot|realman_rm75b` — 机器人本体（默认 dexforce）
- `--hand inspire|linker_l20` — 末端灵巧手；默认 dexforce 配 linker_l20，galbot/realman_rm75b 配 inspire。
  `realman_rm75b` 额外支持 `linker_l20`
- `--scene path/to/scene.xml` — 覆盖默认场景（默认按 `--robot`/`--hand` 选择）
- `--hand-config path/to/config.yaml` — 指定手部重定向配置文件

实物专用:
- `--dexforce-ip 192.168.1.50` — DexForce 机器人 IP
- `--hand-config path/to/config.yaml` — 指定 linker_l20 手部重定向配置文件

## 工具脚本

```bash
# 可视化 MuJoCo 场景（无需 VR）
python3 example/viz/visualize.py dexforce
python3 example/viz/visualize.py realman_l20
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
  - `teleop_real.py` — 按 master 的 RM65 控制循环适配 RM75 双臂；`WristTracker` → 笛卡尔平滑 → `solve_pose_ik` → `RealManOutput`，不调用旧调试保护循环或查看器。
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
