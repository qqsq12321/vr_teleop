# VR Teleop

基于 VR 手部追踪的机械臂遥操作系统。支持两种输入设备：
- **Meta Quest 3** — 配合 [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) 应用，通过 UDP 传输
- **Apple Vision Pro** — 配合 [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用，通过 [avp_stream](https://github.com/Improbable-AI/VisionProTeleop) (gRPC) 传输

## 支持的机器人配置

| 命令 | 机器人 | 模式 | 说明 |
|------|--------|------|------|
| `teleop_sim.py --robot kinova_gripper` | Kinova Gen3 + Robotiq 2F-85 | 仿真 | MuJoCo 仿真，右手控制手臂+捏合控制夹爪 |
| `teleop_real.py --robot kinova_gripper` | Kinova Gen3 + Robotiq 2F-85 | 实物 | Kortex SDK 控制实物机械臂和夹爪 |
| `teleop_sim.py --robot kinova_wuji` | Kinova Gen3 + Wuji Hand | 仿真 | MuJoCo 仿真，右手控制手臂+手部重定向 |
| `teleop_real.py --robot kinova_wuji` | Kinova Gen3 + Wuji Hand | 实物 | Kortex SDK 控制手臂 + Wuji 灵巧手 |
| `teleop_sim.py --robot piper` | Piper 单臂 | 仿真 | 右手控制机械臂+捏合控制夹爪 |
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

### PC 端

- Python 3.10+，conda 环境 `teleop`（`conda activate teleop`）
- MuJoCo (`pip install mujoco`)

### 第三方仓库（位于 `third_party/`）

| 仓库 | 用途 | 来源 |
|------|------|------|
| `AnyDexRetarget` | 手部关键点→灵巧手关节角重定向 | `https://gitee.com/gx_robot/AnyDexRetarget.git` |
| `Kinova-kortex2_Gen3_G3L` | Kinova Gen3 Kortex SDK Python 示例 | `git@github.com:qqsq12321/teleop.git` |
| `mujoco_menagerie` | MuJoCo 机器人模型（Piper, Kinova, Aloha 等） | `git@github.com:wengmister/mujoco_menagerie.git` (git submodule) |

### 额外依赖（按功能）

- **Wuji Hand 实物控制**: `wujihandpy` — Wuji Hand 硬件 SDK
- **Kinova 实物控制**: `kortex-api` (`pip install kortex-api`)

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

# Aloha 双臂（仿真）
python example/teleop_sim.py --robot aloha --input-source avp --avp-ip 192.168.5.32
```

> 实物模式下，双手捏合保持 3 秒可安全停止程序。

## 可选参数

输入源:
- `--input-source quest3|avp` — 输入设备（默认 quest3）
- `--avp-ip <IP>` — Apple Vision Pro IP 地址（仅 avp 模式）

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
- `--hand-config path/to/config.yaml` — 指定手部重定向配置文件（kinova_wuji）
- `--hand-side left|right` — 手部侧向（kinova_wuji，默认 right）

实物专用:
- `--kinova-ip 192.168.1.10` — Kinova IP
- `--kinova-username admin` — Kinova 用户名
- `--kinova-password admin` — Kinova 密码
- `--disable-arm` — 仅控制手部（kinova_wuji，不连接 Kinova 臂）
- `--disable-hand` — 仅控制机械臂（kinova_wuji，不连接 Wuji 手）

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
├── example/                     # 遥操作入口脚本
│   ├── teleop_sim.py               # 仿真遥操作（MuJoCo viewer）
│   ├── teleop_real.py              # 实物遥操作（Kortex SDK）
│   └── scene/                      # MuJoCo 场景文件 (XML)
├── util/                        # 核心模块
│   ├── ik.py                       # 逆运动学求解器
│   ├── quaternion.py               # 四元数运算 + VR→机器人坐标变换
│   ├── udp_socket.py               # UDP 收发 + Quest 数据包解析
│   ├── avp_input.py                # Apple Vision Pro 输入适配器（avp_stream 封装）
│   ├── wrist_tracker.py            # 腕部残差跟踪（EMA 平滑 + deadband）
│   ├── hand_retarget.py            # 灵巧手重定向（landmarks → 关节角）
│   └── arm_move_home.py            # Kinova 回 Home 工具脚本
├── viz/                         # 可视化
│   └── visualize.py                # MuJoCo 场景查看器
├── third_party/                 # 第三方依赖
│   ├── AnyDexRetarget/             # 手部重定向库
│   ├── Kinova-kortex2_Gen3_G3L/    # Kinova Kortex SDK
│   └── mujoco_menagerie/           # MuJoCo 机器人模型
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
