# RealMan RM75-B + Inspire RH56DFX 真机接入

## 1. 日常启动

```bash
conda activate vr_teleop
python example/teleop_real.py --enable-hands
```

**连接真机，无交互确认；默认双臂，上述命令同时启用双手。** 只控制双臂则去掉
`--enable-hands`。原命令 `python example/teleop_real.py --arm both --enable-hands` 仍有效；
`--arm left|right` 保留为可选单臂控制。
**程序先让选中的机械臂运动到 RM75 预设初始姿态，不等待 Pico 追踪。** 默认先左臂、再右臂，
每侧均等待 SDK 的阻塞运动完成；两侧完成后重新读取实测角，首帧腕姿以这个到位后姿态为基准。
启动期间不接入 Pico 控制、不下发手指目标。不自动上电、清故障、编码器校零或设置固定手型；
不控制头、底盘和升降。

Pico 默认使用本机 relay `127.0.0.1:63902`。先在 Pico 端运行手部追踪发送应用，
在另一个终端启动（已经运行则不要重复启动）：

```bash
conda activate vr_teleop
python example/pico4_daemon.py
```

网络和 Pico 模式在 `config/realman_real.yaml`：左臂 `192.168.10.18`、右臂 `192.168.10.19`，
当前 SDK 固定 TCP `8080`。`input.mode: direct` 可不用 relay，此时不要同时让 daemon 占用输入端口。
**本版仅完成离线测试，未进行真机运动验收。** 运行前确认关节序号/零位/方向、硬件使能、
工作区和实体急停；不要同时运行其他机器人控制程序。

## 2. 与主分支对应的控制流程

参考本地 `master`（提交 `366fd12`，2026-07-30）中
`example/teleop_real.py` 的 `_run_rm65_teleop` / `_rm65_send_canfd`。
日常入口现在自己实现该流程，**不再调用 `teleop_realman_real.py` 的旧保护循环**。

```text
启动：连接/读取故障状态 → Movej_Cmd 到 RM75 初始姿态（阻塞，逐臂）
                        → 重新读取两臂实测角 → 模型 FK → Pico 首帧对齐

Pico getter → WristTracker → 笛卡尔平滑 → solve_pose_ik
                                             ↓ 各臂七关节，rad → deg
                              Movej_CANFD(q_deg, follow=True)

Pico 21点 → AnyDex 原始12维 → 六通道映射 → Set_Hand_Angle(..., block=False)
```

| 项目 | 主分支 RM65 与当前日常入口 |
|---|---|
| 初始化规划运动 | `v=20, r=0, trajectory_connect=0, block=True`（对应 master 的 `connect=0, block=1`） |
| 位移倍率 / EMA | `1.5` / `0.5` |
| 腕位置 / 旋转 deadband | `0.03m` / `8°` |
| 笛卡尔位置 / 旋转增益 | `0.08` / `0.08` |
| `solve_pose_ik` | `rot_weight=1.0, damping=0.001, current_q_weight=0.3, skip_tail_joints=0` |
| 下一次 IK 初值 | 上一次 IK 解，不每轮同步读取 SDK 反馈 |
| CANFD | 角度制、`follow=True`，不再追加关节速度/加速度/步长限制 |
| 循环节拍 | 目标 `20ms`；工作不足时补足休眠，超时不补发、不按固定 `100ms` 退出 |
| 无有效 getter 结果 | `250ms` 后跳过该臂下发，不结束会话；恢复后沿用原腕基准 |
| 手指下发 | 每轮重发最近有效目标，无额外 `10Hz` 上限、去重或 `±20` 步长限制 |

**输入语义也按主分支，而不是之前的严格新帧检查：** 每轮调用 Pico 的普通 getter，
同侧有效腕姿或有效手部重定向结果均刷新该臂的有效时间。getter 可能持续返回缓存，
因此 `250ms` **不代表能检测网络断流或真实追踪过期**。超时分支只跳过机械臂命令，
手指已有目标仍会下发；不调用 `Move_Stop_Cmd`，也不重新对齐恢复后的腕姿。

保留的必要硬件差异：
- RM65 单臂六关节改为 RM75 双臂各七关节，用 `left_tcp` / `right_tcp` site 做 IK，
  `dof_indices` 仅开放对应臂的七关节；其他臂、头和手指不是该次 IK 的求解变量。
- 启动运动采用 RM75 的七轴 home，不能复制 RM65 六轴目标。动作完成后重新读取实测姿态，
  将其用于初始末端位姿和 IK 的 `home_qpos`，不把下发目标或运动前反馈当作到位测量。
- 当前现场使用 SDK 4.3.7 的 `Movej_CANFD(joint, follow, expand=0)`，
  主分支使用 API2 的 `rm_movej_canfd(q_deg, follow=True)`；当前不使用额外轨迹模式参数。
- 灵巧手沿用现场 RealMan 末端接线和 `Set_Hand_Angle` 后端，不改成主分支的电脑直连串口，
  也不复制主分支启动时的固定手型指令。六通道映射与其发送边界换算保持一致。

日常入口只用 MuJoCo 做 FK/IK，**不启动仿真、查看器或物理步进**，不修改已正常工作的仿真入口。
名义 `50Hz` 不是实测保证；两次 IK、重定向和串行 SDK 调用的耗时决定实际循环频率。

### 初始姿态的位置与参数

目标来自 `utils/robots/realman_rm75b.py`，与已经使用的仿真 home 共用常量，**没有修改仿真值**。
下面按各臂 SDK 的 J1…J7 顺序，单位为 **弧度**；适配器发送前转换为角度：

```python
LEFT_ARM_HOME_QPOS = (-0.47, 0.0, -0.50, -2.04, 0.62, 1.09, -0.59)
RIGHT_ARM_HOME_QPOS = (0.47, 0.0, 0.50, 2.04, -0.62, -1.09, 0.59)
```

启动打印 `Moving left/right to initial pose (deg): ...` 后，通过 SDK
`Movej_Cmd(q_deg, v=20, r=0, trajectory_connect=0, block=True)` 执行规划运动。
`20` 是主分支启动运动使用的速度百分比，仅用于初始化；后续仍是 `Movej_CANFD(..., follow=True)`，
没有新增遥操限速。无需增加命令行参数或口令。

有手部配置时先检查重定向器可加载，再允许初始运动。任一初始化调用报错、被中断，或到位后读取
发现控制器故障，均结束会话并尽力停止两臂；不会继续 CANFD 或手指控制。
这些模型目标和从现场当前姿态出发的运动路径尚未做真机验收，启动前应核对实物零位/方向和路径空间。

### 配置与退出

`control` 段为上表参数；日常入口不使用 YAML 的 `safety` 和 `hands.rate_hz`，
也不调用 `SafetySupervisor` / 模型碰撞检查。原先的 `150ms` 新帧报错、`100ms` 循环报错
和额外关节低速整形不在本入口中。SDK 连接/型号/启动故障、目标维数/有限值检查仍保留；
SDK 返回错误结束会话，不自动重试。

Ctrl+C、SIGTERM 或异常退出时尽力对两臂执行停止，再关闭连接；不会自动松手、回零或断电。
没有独立的硬实时硬件看门狗；SDK 阻塞、进程崩溃或网络故障时无法保证停机，
需使用已验证的控制器断流策略和实体急停。停止手臂不等于取消灵巧手最后目标。

## 3. 机器人开机和网络参考

依据用户提供的《双臂机器人平台使用手册V1.3》，下面的页码均为 **PDF阅读器页序**：
- §6.1，PDF第17页：平台各模块IP。
- §6.2，PDF第18–19页：总电源开关、急停复位、状态灯、关机。
- §6.3，PDF第20–23页：对外以太网接口、电脑IP、Web示教器登录。
- PDF第24–25页：末端电源与夹爪操作；其中“两指夹爪”操作不能直接当作Inspire灵巧手配置。
- PDF第27页：RM75的5–7关节无抱闸，断电/急停时需防止末端及负载在外力下运动。

**机器人尚未开机时**，先确认装配和末端线缆连接正确、工作区无人、无负载风险，
然后按手册操作：

1. 总电源开关在移动底盘中间中空位置的上侧，位置见手册图12。
   长按，听到连续提示声后松开。已经开机时不要重复长按，长按也用于关机。
2. 如急停处于按下状态，先确认触发原因已排除且允许恢复，
   再按手册顺时针旋转复位。手册描述正常开机后胸前状态灯为绿色（图13）。
   绿灯不是本项目的运动授权，也不能替代示教器状态检查。
3. 电脑用网线连接机器人背部下方的对外以太网接口（图15、16）。
   手册推荐电脑有线IP `192.168.10.33`；必须确认地址未占用，
   不要使用平台已占用的 `.10`、`.100`、`.18`、`.19`。
   网络参数应与现场机器人一致，不要修改机器人控制器IP来迁就电脑。
4. 浏览器分别访问 `192.168.10.18`（左臂）与 `192.168.10.19`（右臂）。
   手册中的出厂Web登录为账号 `user`、密码 `123`；若已修改，以现场配置为准。
   这不是机器人主控Linux系统的登录账号。
5. 在原厂Web示教器查看两臂状态、告警、关节角及运行模式。
   平台手册第23页把详细机械臂示教操作转引到机械臂/WEB示教器用户手册，
   **没有完整展开各固件版本的关节上电、使能按钮流程**；应依实际示教器和对应手册操作，
   不要仅凭“绿灯”假定已使能，也不要未查明原因就清除告警。


日常入口启动即允许软件下发；旧调试入口的 `--enable-motion` 也仅是软件开关，
两者都不负责机器人总电源、关节上电、硬件使能或真实/仿真模式切换。
启动后不要再由其他控制程序改变初始关节姿态，本入口不会持续刷新实测基线。
头部按模型零位固定，底盘静止；本链路没有头部/底盘反馈。

## 4. Inspire 后端和映射

`hands.backend` 必须显式选择，**禁止写寄存器试探自动识别**：

| 后端 | 下发接口 | 要确认的条件 |
|---|---|---|
| `native`（默认） | `Set_Hand_Angle([6路0..1000], block=False)` | 当前控制器固件确实支持已连接的Inspire手；成功返回不等于实际到位 |
| `modbus` | `Set_Modbus_Mode` + `Write_Registers` | 确認末端port=1、波特率、device_id和寄存器地址适用当前手；独占RS485 |

配置的 Modbus 参数来自本机已有控制节点参考：`115200`，`device_id=1`，
角度写入起始地址 `1486`，6个寄存器，SDK字节数组使用**高字节在前**。
这不同于现有直连串口 Inspire 包的低字节在前，不能混用。
程序不自动修改末端24V供电，也不会为确认连接而写角度。

六通道顺序：**小指、无名指、中指、食指、拇指弯曲、拇指旋转**。
映射索引为 AnyDex 原始12维的 `(4,6,2,0,9,8)`；归一化上限为
`(1.47,1.47,1.47,1.47,0.6,1.308)` rad；当前模型配置中0闭合、1000张开。
这些是项目映射约定，必须先逐通道验证实际方向与限位。

**首版不读取手指实际角度、电流或接触力。** 首次发送前不知道实物手型，不能保证无起始跳变。
必须在外部验证低速低力设置、空手且无人接触后才开启。
日常入口沿用主分支换算顺序：先按 `0..2000` 整数归一化，再逐通道
`round(value / 2)` 转为 native SDK 的 `0..1000`。不先重排成 MuJoCo 手指顺序。
收到无效重定向结果时保留上次目标，每轮重发；没有额外低频/步长整形。
停止下发或停止手臂都不保证手指取消最后目标，也不会自动张手。

## 5. 可选旧调试入口（不是日常真机调用路径）

`example/teleop_realman_real.py` 保留旧的检查、只读和离线预览功能，
**与上述日常入口不是同一控制循环**。这些不是启动日常遥操必须执行的模式。

```bash
# 离线检查配置和模型；不导入 SDK，不创建机器人连接
python example/teleop_realman_real.py --check

# 离线：只接 Pico，不连接机械臂
python example/teleop_realman_real.py --dry-run --arm both --viewer

# 离线追踪失效时冻结，恢复后重新对齐
python example/teleop_realman_real.py --dry-run --arm both --viewer --pause-on-tracking-loss

# 连接两臂但不下发运动、手指、上电或停止命令
python example/teleop_realman_real.py --read-only --duration 3
```

旧入口也保留 `--enable-motion --arm both --enable-hands` 供兼容，但**日常使用第1节命令**。
旧入口的行为与日常入口不同：
- 使用分侧时间戳/新帧策略和 `SafetySupervisor`；配置默认为输入 `250ms`、反馈 `200ms`、
  命令有效期 `80ms`、关节余量 `5°`、跟踪误差 `10°` 持续 `200ms`、碰撞余量 `2cm`。
  固定 `100ms` 循环退出和关节速度/加速度/步长整形已移除。
- 旧入口启用真机运动时，追踪失效跳过本轮后续下发，恢复后沿用原基准；其他保护/SDK错误仍退出。
  默认 dry-run 追踪失效仍退出。`--pause-on-tracking-loss` 仅用于 dry-run，恢复时重新对齐；
  与旧入口真机的恢复方式不同。
- 旧入口的手指最高 `10Hz`，每次变化最多 `20` 通道单位；这些不影响日常入口。
- 模型检查显式计算几何距离，`ncon == 0` 不能证明无碰撞；保守手指包围球和插值采样
  都不是连续碰撞证明，也不覆盖未知环境、线缆、载荷或模型误差。
- `--viewer` 仅在旧入口提供，独立进程显示最新模型姿态；显示不会回写控制状态。
  关闭窗口或显示进程异常会结束调试会话。

旧入口的 `wrist tracking lost/stale` 指输入失活或没有及时处理新帧；
`Pico4 relay connection closed` 出现在退出时也可能只是主动清理，不能单凭此行判定网络先断。
时间戳是本机接收时间，不是设备采集时间。日常入口不采用这一严格新帧策略。

## 6. 离线回归测试

```bash
conda activate vr_teleop
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
# 可选 GUI 退出测试，需要图形桌面；仍不访问机器人
REALMAN_TEST_GUI=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_process_viewer.py
```

`tests/test_teleop_real_cli.py` 使用 fake SDK/Pico 和实际模型 IK，核对本地 master 的参数、
双臂初始 MoveJ 的目标/参数/阻塞顺序、到位后重新读反馈、第一条 CANFD 保持到位后实测角、
初始化失败/中断不进入遥操、双臂七关节 rad→deg 原样发送、IK 前次解初值、手部映射与每轮下发、
超时跳过/恢复、循环节拍、无确认输入及异常清理。其余测试覆盖适配器、输入和旧调试保护逻辑。
所有 SDK 测试使用 fake，不自动访问或驱动机器人；离线通过不替代实物验收。

### Right J1 installation-offset candidate

The daily `example/teleop_real.py` entry now reads
`right_j1_offset_deg: -180.0` from `config/realman_calibration.yaml`.
This is a user-requested candidate, **not a physically verified calibration**.
Startup MoveJ and teleop CANFD both use `SDK J1 = model J1 - 180°`;
feedback uses the inverse `model J1 = SDK J1 + 180°`. Other joints, the left
arm, and simulation assets are unchanged. The startup log displays SDK angles,
so the right startup J1 is approximately -153.07°, not +26.93°.

No modulo wrapping or automatic choice of rotation direction is performed.
Model-space joint limits do not establish hardware feasibility after this offset;
the physical limits, cable clearance and startup sweep need verification before
running. The program still moves to the initial pose before Pico input starts.
Set this configuration value to `0.0` to restore the previous mapping.
The legacy debug entry does not consume this separate calibration file.
