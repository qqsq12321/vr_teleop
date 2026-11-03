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
| `teleop_sim.py --robot realman_rm75b_single` | 单臂 RM75-B（落地单臂）+ 可选 Inspire | 仿真 | `--hand none`（默认）或 `inspire`；**唯一开启胶囊避碰的机器人** |
| `teleop_rm75b_single_real.py` | 单臂 RM75-B + 可选 Inspire | 实物 | RM_API2（**非** 双臂的旧 SDK），默认只做离线检查，需显式 `--enable-motion` |
| `teleop_sim.py --robot cvte` | CVTE CARM D3 双臂 + 平行夹爪 | 仿真 | 捏合距离控制夹爪开合 |
| `teleop_cvte_real.py` | CVTE CARM D3 双臂 + 平行夹爪 | 实物 | 官方 `carm` SDK（WebSocket）→ IK → `track_*_joint`；默认从当前姿态开始 |

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

## 单臂 RM75-B（落地单臂，仿真 + 真机）

立在地上的单条 RM75-B 七轴臂，法兰工具口可选接 Inspire RH56DFX 灵巧手。
**与双臂路径完全分开**：自己的配置 `config/realman_rm75b_single_real.yaml`、
自己的场景，没有双臂那套 J1 零位补偿。真机走 **RM_API2**
（`pip install Robotic_Arm`），不是 `third_party/realman_api` 里的旧 SDK，两者不通用。

### 仿真

```bash
conda activate vr_teleop
python example/pico4_daemon.py                                    # 另一个终端，relay 模式需要
python example/teleop_sim.py --robot realman_rm75b_single --hand inspire --input-source pico4
```

场景由 `tools/build_realman_rm75b_single_scene.py --hand {none,inspire,all}` 生成，
已入库，重建不需要厂商 ROS 包。

### 真机

四种模式，按安全程度排序，**默认是最安全的 `--check`**：

```bash
# 1. 离线检查：不开任何 socket，校验初始姿态在限位内，打印将要下发的 SDK 角度
python example/teleop_rm75b_single_real.py --check

# 2. 只读：连上机械臂打印实测关节角，不使能、不下发运动
python example/teleop_rm75b_single_real.py --read-only --duration 10

# 3. 只驱动灵巧手：机械臂连上但不使能，用于手第一次上电
python example/teleop_rm75b_single_real.py --hand-only --hand inspire

# 4. 完整遥操（机械臂会动）
python example/teleop_rm75b_single_real.py --enable-motion --hand inspire
python example/teleop_rm75b_single_real.py --enable-motion --controller   # 手柄，握把键为死手开关
```

新装机器先跑 1、2 确认关节角正常，手第一次接上先跑 3，再放开到 4。

手势模式下使能后要等追踪数据齐备，然后 3-2-1 倒数，倒数结束的那一帧作为位姿基准，
并按头显朝向对齐"前方"，所以站位和朝向换了也不用重新标定。
工具口电压**只读不设**，读到 0V 会拒绝启动（在示教器里设置）。
`--controller` 模式不驱动灵巧手。

### 自碰撞与地面避让

**只有这台机器人开启**，其他机器人的 IK 一行未变（默认关闭时输出逐位相同）。
实现在 `utils/core/capsule_collision.py`：每节连杆拟合一个胶囊，手拟合手掌和拇指两个，
地面是一个水平半空间。

- **软避让**：胶囊表面间隙进入 **5 cm** 后，IK 里加一个渐进弹簧把它们推开。
  刚进入时几乎不推，越近越硬，所以是"让一下"而不是撞墙。
- **硬兜底**：真机上若某一帧仍会越过 **5 mm** 间隙，就沿这一帧的动作二分到刚好 5 mm 停下，
  而不是整帧丢弃，机械臂会沿障碍滑开而不是卡死。只有已经在 5 mm 以内还要往更深处走才原地不动。
  终端打印 `Limiting:` 或 `Holding:`，最多每秒一次。
- **关节限位**：解出的角度统一留 3° 余量（`mj_integratePos` 不会自己截断转动关节）。

胶囊按每节最粗处拟合，所以比实际零件胖（金属填充 55%–60%，手只有 26%–38%）。
实测胶囊间隙 5 mm 时真实金属还隔 33–38 mm：**偏保守，不会漏判**，代价是部分紧凑姿态会被提前推开。
链上间隔小于 3 的配对不检查——相邻连杆共用轴承本来就贴着，间隔 2 的穷举扫描显示关节限位
已经让它们至少隔 17 mm。

地面高度由 spec 的 `collision_floor_z` 决定，默认 `0.0`（基座坐在地上）。
装到底座或台子上要改这个值，设 `None` 关掉地面检查。
软硬程度调 `utils/core/capsule_collision.py` 的 `weight`（默认 5，越大越硬）和 5 cm 影响距离。

**看效果**（都不是运行机器人必须的）：

```bash
# 不用 VR：自动折向自身或压向地面，循环回放，画出胶囊；加 --no-avoid 对比
python tools/view_capsule_avoidance.py --hand inspire
python tools/view_capsule_avoidance.py --hand inspire --goal 0.5 0 -0.1 --no-avoid

# 遥操时实时看胶囊（参数同对应入口）
python tests/teleop_sim_capsules.py --hand inspire --input-source pico4
python tests/teleop_real_capsules.py --enable-motion --hand inspire
```

颜色：绿=还远，橙=IK 正在推开，红=到硬线、这一帧不下发。
仿真脚本可加 `--no-avoid` 关掉避让对比；真机脚本**故意没有**这个开关。

## CVTE CARM D3 双臂（仿真 + 真机）

固定基座的七轴人形双臂（每臂 7 关节）+ 每侧一个单自由度平行夹爪，无躯干/头部。
URDF 来自厂商 `carm_d3` 包，产品资料见厂商手册与 [carm_demo](https://github.com/cvte-robotics/carm_demo)。

### 仿真

```bash
conda activate vr_teleop
python example/teleop_sim.py --robot cvte --input-source avp --avp-ip <Vision Pro IP>
python example/teleop_sim.py --robot cvte --input-source pico4          # Pico relay，需先开 daemon
python example/viz/visualize.py cvte                                     # 只看模型
```

- 初始姿态：大臂下垂、肘部 90°、小臂和夹爪朝前（`utils/robots/cvte_carm_d3.py` 的 `ARM_HOME_QPOS`）。
- 夹爪由拇指-食指捏合距离控制：`0.02m` 全闭，`0.10m` 全开（`TeleopArmController.update_hand_from_pinch`，
  只对单自由度夹爪生效，灵巧手机器人不受影响）。
- 场景 `example/scene_config/scene_cvte_carm_d3.xml` 由 `tools/build_cvte_carm_d3_scene.py` 从
  `assets/arm_body/cvte_carm_d3/source/urdf/carm_d3.urdf` 生成；URDF 的 `<mimic>` 夹爪联动由场景里的
  `<equality>` 补上（MuJoCo 导入 URDF 时不会转换 mimic）。细节见 `assets/arm_body/cvte_carm_d3/README.md`。

### 真机

前置：
- 安装官方 SDK 到 `vr_teleop` 环境：`pip install carm`（纯 Python，依赖 `websocket-client`、`zeroconf`）。
- 控制器默认 IP `10.42.0.101`，PC 网口需配 `10.42.0.x`，例如
  `sudo ip addr add 10.42.0.201/24 dev <网口>`；能打开 `http://10.42.0.101` 网页上位机即为连通。
- 急停按钮必须接上才能上电。

推荐上手顺序：

```bash
conda activate vr_teleop

# 1. 只读检查：不使能伺服、不下发任何指令；打印关节、夹爪、FK 与控制器法兰位姿对比
python example/teleop_cvte_real.py --read-only --report

# 2. 移到初始姿态后退出（不连 VR）
python example/teleop_cvte_real.py --home-only

# 3. 单臂遥操，再双臂
python example/teleop_cvte_real.py --arm left --input-source avp --avp-ip <Vision Pro IP>
python example/teleop_cvte_real.py --input-source avp --avp-ip <Vision Pro IP>
python example/teleop_cvte_real.py --input-source pico4                  # Pico relay
```

行为说明：
- 默认**从机械臂当前姿态**开始遥操；`--home` 先在 `--home-duration`（默认 6s）内平滑移到初始姿态。
  零位（双臂垂直）是奇异位形，建议先 `--home`。
- 启动运动时调用 SDK `set_ready()`（清错、伺服使能、位置模式），两臂都会使能。
- 关节角原样下发，无偏移/符号转换：厂商 `joints_bind.json` 一一对应，且已在实物上用 `--report`
  验证模型 FK 与控制器法兰位姿相差约 1.5–1.8 mm。
- 回 home 不用 SDK 的定时 `move_*_joint`：该控制器（固件 1.0.260819）拒绝其 `TASK_PVT` 任务，
  因此 home 和遥操共用同一条 `track_*_joint` 流式通道。
- 夹爪：仅当控制器报告末端为 `gripper`（`get_*_eeff_type`）时启用捏合控制，开口度映射到控制器的
  `eeff_lower..eeff_upper`（米）。末端为 `flange` 时只控制手臂；装夹爪后需在网页上位机切换工具并重启。
- **D3 关节没有抱闸**：伺服下使能、硬件急停或断电都会使手臂下坠。程序退出时只调用 `task_stop()`，
  伺服保持使能、手臂停在原位；程序永不调用 `set_servo_enable(False)` / `emergency_stop()`。
- 停止：`Ctrl+C`，或 AVP 左手握拳 3 秒。

安全保护（任何一项触发即停止并退出）：关节反馈超过 `--feedback-timeout`（0.5s）未更新、控制器报错
（3001 警告除外）、实测关节持续偏离指令（0.3 rad 超过 0.5s）。指令始终限制在关节限位内
（`--joint-margin` 0.05 rad）并按 `--max-joint-speed`（1 rad/s）限速。VR 超过 0.25s 无新数据时停止下发，手臂保持原位。

手感参数（默认值）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--position-scale` | 1.0 | 手腕平移 1:1 映射到夹爪 |
| `--position-deadband` / `--rotation-deadband` | 0 / 0 | 死区，0 表示关闭（与仿真一致） |
| `--pos-gain` / `--rot-gain` | 0.3 / 0.3 | 每帧向目标走的比例，90% 跟上约 0.13s；1 = 不平滑 |
| `--ema-alpha` | 0.8 | 腕部平移 EMA |
| `--publish-hz` | 50 | 下发频率 |

运行时每秒打印诊断：手部位移、IK 目标位移、实测夹爪位移（cm），以及每个关节"上一帧指令 − 实测"（度）。
可加 `--viewer` 打开只读 MuJoCo 窗口显示实测姿态。

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
- **CVTE CARM D3 实物**: `carm`（`pip install carm`），装在 `vr_teleop` 环境；仿真和单元测试不需要它。

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

`pico4_daemon.py` 与 C++ 版 `pico_relay_daemon` 使用同一组端口（TCP 63901/63902），只能运行其中一个；
任一个在运行时，`--pico4-mode relay` 都能直接使用。启动报 `address already in use` 时，用
`ss -ltnp | grep -E ':(63901|63902)\b'` 或 `ps aux | grep pico` 找到已在运行的那个。

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
- `--position-scale` — 手腕位移映射倍率（默认 1.0，即 1:1）
- `--ema-alpha 0.8` — EMA 平滑系数
- `--rot-weight 1.0` — IK 旋转权重
- `--ik-damping 0.001` — IK 阻尼系数
- `--ik-current-weight 0.1` — IK 当前姿态权重

仿真专用:
- `--robot dexforce|galbot|realman_rm75b|cvte` — 机器人本体（默认 dexforce）
- `--hand inspire|linker_l20|gripper` — 末端执行器；默认 dexforce 配 linker_l20，galbot/realman_rm75b 配 inspire，
  cvte 配 gripper（单自由度平行夹爪，由捏合距离驱动）。`realman_rm75b` 额外支持 `linker_l20`
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
python3 example/viz/visualize.py cvte
# 或直接指定 XML
python3 example/viz/visualize.py example/scene_config/scene_dexforce.xml

# 单臂 RM75-B 碰撞避让回放（无需 VR，见上文"自碰撞与地面避让"）
python tools/view_capsule_avoidance.py --hand inspire
```

## 项目结构

```
vr_teleop/
├── utils/                       # 核心算法、适配器、机器人规格
│   ├── core/
│   │   ├── types.py               # 共享数据结构
│   │   ├── quaternion.py          # 四元数运算 + 坐标变换
│   │   ├── body/
│   │   │   ├── ik.py              # 机械臂/本体 IK（可选碰撞避让、关节限位余量）
│   │   │   └── wrist_tracker.py   # 腕部残差跟踪
│   │   ├── capsule_collision.py   # 胶囊自碰撞 + 地面模型（仅单臂 RM75-B 启用）
│   │   └── hand/
│   │       └── retarget.py        # 灵巧手重定向
│   ├── adapters/
│   │   ├── input/
│   │   │   ├── quest3.py
│   │   │   ├── avp.py
│   │   │   └── pico4.py
│   │   └── output/
│   │       ├── mujoco_sim.py
│   │       ├── dexforce_real.py
│   │       └── cvte_real.py       # CVTE D3 carm SDK 适配（限位、限速、故障锁存）
│   └── robots/
│       ├── dexforce.py            # DexForce 关节映射、home pose、配置路径
│       ├── cvte_carm_d3.py        # CVTE D3 关节名、home pose、控制器 IP
│       └── realman_rm75b_single.py # 单臂 RM75-B 关节名、home pose、碰撞胶囊链
├── assets/                      # 机器人/物体模型资产
│   ├── arm_body/dexforce/          # DexForce 本体 mesh、贴图、URDF
│   ├── arm_body/cvte_carm_d3/      # CVTE D3 厂商 URDF + STL
│   ├── dex_hand/linker_l20/        # linker_l20 手部 URDF/MJCF、visual/collision mesh
│   └── objects/                    # 示例物体 mesh 和预览图
├── example/                     # 遥操作入口和示例配置
│   ├── teleop_sim.py
│   ├── teleop_real.py
│   ├── teleop_cvte_real.py         # CVTE D3 真机入口
│   ├── teleop_rm75b_single_real.py # 单臂 RM75-B 真机入口（RM_API2）
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
