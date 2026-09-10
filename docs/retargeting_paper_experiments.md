# AnyDexRetarget 论文实验接入（离线优先）

本说明只描述采集与评测接入点，不修改控制器、不启动设备、不解锁运动。
现有 RealMan / Pico / Inspire 开发改动保持原样。真机操作仍须遵守对应硬件的安全验收流程。

## 两仓库分工

- `../AnyDexRetarget`：算法、冻结研究配置、离线 runner、消融、论文初稿。
- `vr_teleop`：输入设备适配、手臂/手控制、系统部署与真实任务实验。
- 论文算法效果必须通过相同输入下的配对回放及受控真实任务验证，不能以支持设备数量代替。

主工作包：`/home/qsq/qqsq12321/AnyDexRetarget/paper/README.md`。
离线工具说明：`/home/qsq/qqsq12321/AnyDexRetarget/research/README.md`。

## 先核对实际加载的代码与配置

本仓库有 `third_party/AnyDexRetarget`，不等于并列主仓库的修改版本。
`utils/core/hand/retarget.py` 默认配置也来自 third_party。仅设置 PYTHONPATH
不保证能覆盖入口脚本后续的 `sys.path.insert`。在实验进程中、初始化前确认：

```python
import anydexretarget
print("AnyDex implementation:", anydexretarget.__file__)
print("Selected experiment config:", config_path)
```

研究回放从主仓库根目录执行；将源码 hash、配置 hash、URDF hash 写进 manifest。
正式在线对比需显式选择一致的研究配置，并记录实际运行版本；不要在生产入口中
未经测试地替换 third_party 或默默修改已调好的硬件配置。

## 原始数据的接入点

查看 `utils/core/hand/retarget.py`：

1. `landmarks_to_mediapipe` 将 Unity 坐标的 63 floats 转为 RH `(21,3)` 点。
2. `HandRetargeter.retarget_mediapipe` 将点交给 `self._retargeter.retarget(...)`。
3. 论文记录点位于 **坐标约定转换之后、Retargeter 的 wrist-frame 变换和滤波之前**。
   这里的 `raw_keypoints` 指相对于 Retargeter 尚未变换，而非未经解析的网络包。
4. 已经提供 RH 点的输入，不能再次调用 Unity 转换；所有点必须确认使用米和 MediaPipe 顺序。

可使用主仓库中的 `research.recording.TrialRecorder`；下面是接入示意，**不是已接好的
在线录制命令**。应在已有采集会话中选择新鲜有效帧，避免重复记录同一个网络包：

```python
from research.recording import TrialRecorder

recorder = TrialRecorder(
    subject_id="P01", trial_id="pinch_index_01",
    input_source="quest3", hand_side="right",
)
# 在已有输入处理代码中，对每个新鲜、有效、单位正确的样本调用：
# recorder.append(mediapipe_pts, acquisition_timestamp_s)
# 试次结束后，在非控制线程中保存，不覆盖已有文件：
# recorder.save("/home/qsq/qqsq12321/AnyDexRetarget/research/data/P01/pinch_index_01.npz")
```

要求主仓库根目录在该离线/采集进程的导入路径中。该 helper 不会连接设备或机器人；
尚未实现 UI 快捷键、输入事件接线和跨设备时钟同步。不要把保存压缩 NPZ 的操作放进
实时控制回调中。

- 时间戳使用明确的采集时钟，单位秒；严格递增。不用控制循环时间冒充传感器采样时间。
- 不按墙钟重新戳记缓存帧；记录丢帧、失跟踪、设备帧号和时钟定义的 sidecar 信息。
- 接触标签默认为未知 `-1`；可提供 4 个独立标注，顺序 index/middle/ring/pinky。
  不能用算法自己的 alpha 生成标签再宣称接触识别成功。
- 真实任务还要单独记录物体、成功判定、急停、试次顺序和同步视频；NPZ 不代替这些记录。
- 同意与隐私：用户编号去标识化，原始人体数据不提交 Git。

## 冻结配置与实验公平性

现有研究快照是 Quest3 pilot，不是已验证的 Pico/AVP 最优配置。其他输入需要独立冻结
与开发集验证，禁止把配置改名就当成跨设备免标定证据。详见主仓库的
`paper/EXPERIMENT_PROTOCOL.md`。

当前本仓库只新增本文档；算法回放、原始数据 helper 与论文工具在主仓库。
兼容性测试只证明接口/关节顺序等软件约束，**不代表完成真机安全验收**。
