# Hand Tracking Streamer - PC 端环境配置指南

## 要装环境的时候直接将这个文档扔给ai就行了

本文档用于在 PC 端配置 Hand Tracking Streamer 的接收与可视化环境。

## 1. 前置条件

- Linux 系统（已在 Ubuntu 22.04 上验证）
- 网络连接（用于下载依赖）
- Meta Quest 头显已安装 Hand Tracking Streamer 应用

## 2. 创建 conda 环境

```bash
# 首次使用需接受服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建环境
conda create -n teleop python=3.10 -y

# 激活环境
conda activate teleop
```

## 3. 安装依赖

```bash
# 安装 git（如系统已有 git 可跳过）
conda install -y git

# 克隆仓库
git clone https://github.com/wengmister/hand-tracking-streamer.git ~/hand-tracking-streamer

# 安装 Python SDK 和可视化依赖
pip install hand-tracking-sdk matplotlib numpy
```

## 4. 开放防火墙端口

```bash
# 如果使用 ufw
sudo ufw allow 8000/tcp
sudo ufw allow 9000/tcp
sudo ufw allow 9000/udp

# 如果使用 iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9000 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 9000 -j ACCEPT
```

## 5. 运行可视化脚本

```bash
conda activate teleop

# 查看本机 IP
hostname -I

# 启动可视化（TCP 模式，推荐）
python ~/hand-tracking-streamer/scripts/visualizer.py \
    --protocol tcp \
    --host <本机IP> \
    --port 8000 \
    --show-fingers

# 或使用 UDP 模式（低延迟）
python ~/hand-tracking-streamer/scripts/visualizer.py \
    --protocol udp \
    --host 0.0.0.0 \
    --port 9000
```

## 6. Quest 头显端配置

1. 确保 Quest 头显与 PC 在**同一局域网**下（Wi-Fi 和有线均可，只要连同一路由器）
2. 打开 Hand Tracking Streamer 应用
3. 设置连接参数：
   - **IP**: PC 的局域网 IP（通过 `hostname -I` 查看）
   - **Port**: 与脚本启动时一致（如 `8000`）
   - **Protocol**: 与脚本启动时一致（如 `TCP`）
4. 开始追踪，PC 端即可看到手部关键点可视化

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| Quest 报 `connection error: access denied` | 检查防火墙是否已开放对应端口 |
| 无法连接 | 确认 PC 和 Quest 在同一局域网，尝试从 Quest ping PC 的 IP |
| 路由器 AP 隔离 | 部分路由器会隔离 Wi-Fi 和有线设备，需在路由器设置中关闭 AP 隔离 |
| `ModuleNotFoundError` | 确认已激活 `teleop` 环境：`conda activate teleop` |

## 已验证的软件版本

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| hand-tracking-sdk | 1.1.0 |
| matplotlib | 3.10.8 |
| numpy | 2.2.6 |
