# mujoco_vla_project

## 项目简介

本项目用于完成以下链路：

- 在 MuJoCo 中搭建 G1 上半身任务场景
- 实现上半身末端目标位置跟随（当前为右手）
- 打通 RGBD 相机读取与数据录制
- 为后续 XR 遥操接入、数据采集与 benchmark 做准备

当前代码状态：

- **阶段 1 已完成**：双手末端连续轨迹跟随 + RGBD + 同步存储
- **阶段 2 已打通**：`PICO controller/hand -> TeleVuerWrapper -> bridge -> MuJoCo target -> G1 双手末端跟随`
- **阶段 4 已建立骨架**：可对录制 episode 做基础 tracking benchmark，ACT / DP3 / GR00T runner 仍是待接入占位

## 环境要求

- Ubuntu 22.04（推荐）
- Python 3.10
- MuJoCo 3.x
- Conda（推荐）
- 可选：PICO / Quest 等支持 WebXR 的 XR 设备

## 安装步骤

```bash
conda create -n mujoco_vla python=3.10 -y
conda activate mujoco_vla

pip install --upgrade pip
pip install mujoco glfw numpy scipy matplotlib opencv-python imageio tqdm
```

真实 XR 接入还依赖 `xr_teleoperate` 的 `televuer` 模块。当前默认路径为：

```text
/home/wll/xr_teleoperate
```

如果 `mujoco_vla` 环境没有安装 `vuer/televuer`，代码会临时复用本机已有 `tv` 环境中的 Python 3.10 依赖：

```text
/home/wll/miniconda3/envs/tv/lib/python3.10/site-packages
```

可选（Ubuntu 下离屏渲染常用依赖）：

```bash
sudo apt update
sudo apt install -y libgl1-mesa-glx libglfw3 libglew-dev libosmesa6-dev libxrender1 libxext6 libx11-6
```

## 快速开始（阶段 1 + 阶段 2 bridge）

### 1. 阶段 1 当前支持能力

- 双手连续轨迹跟随
- RGBD 读取与保存（`rgb + depth + camera intrinsics/extrinsics + timestamp`）
- 同步记录：`joint_state/action/target_eef/actual_eef/tracking_error/timestamp`

### 2. 运行阶段 1（可视化窗口）

```bash
python scripts/run_stage1.py --duration 10 --save-video
```

或使用脚本入口：

```bash
bash scripts/record_data.sh 10 data/raw
```

### 3. 录制双手连续轨迹 sample episode

```bash
python scripts/record_stage1_bimanual_trajectory.py
```

### 4. 运行阶段 2 bridge demo（无真实 XR）

`teleop` 模式（推荐，mock XR 输入）：

```bash
python scripts/teleop_mujoco_demo.py --mode teleop
```

右手单手稳定模式（推荐先跑这个）：

```bash
python scripts/teleop_mujoco_demo.py --mode teleop --hand-mode right_only
```

说明（默认策略）：

- `teleop` 模式采用“跟手优先”默认：优先提升控制带宽，而不是在 bridge 层强行限速目标。
- bridge 默认仅做：坐标变换、缩放、工作空间裁剪、轻度低通滤波。
- `configs/bridge.yaml` 中 `max_target_speed` 默认 `0.0`（关闭目标限速）。

`trajectory` 模式（沿用阶段 1 目标生成）：

```bash
python scripts/teleop_mujoco_demo.py --mode trajectory
```

`trajectory` 也支持右手单手模式：

```bash
python scripts/teleop_mujoco_demo.py --mode trajectory --hand-mode right_only
```

调快手臂响应（可选）：

```bash
python scripts/teleop_mujoco_demo.py --mode trajectory --max-joint-speed 1.5 --task-gain 3.5
```

`teleop` 模式下，脚本内部默认会使用更高控制带宽（无需手动传参）以提高遥操作跟随性。

### 5. 接入真实 PICO / xr_teleoperate

先确认 `xr_teleoperate` 已安装，默认路径假设为：

```bash
/home/wll/xr_teleoperate
```

运行本仓库的 MuJoCo demo 时，不要同时启动下面这些程序，否则它们会抢 Vuer 默认端口 `8012`：

- `xr_teleoperate/teleop/teleop_hand_and_arm.py`
- `unitree_sim_isaaclab/sim_main.py`

本仓库会自己创建 `TeleVuerWrapper`，链路是：

```text
PICO controller/hand -> Vuer/TeleVuerWrapper -> adapter_xr -> bridge -> MuJoCo target -> G1 hand
```

#### 5.1 先验证 mock XR 右手

```bash
python scripts/teleop_mujoco_demo.py \
  --mode teleop \
  --xr-source mock \
  --hand-mode right_only \
  --duration 20 \
  --output-dir data/samples/stage2_right_only_mock
```

通过标准：MuJoCo 中红球运动，机器人右手能跟随红球。

#### 5.2 真实 PICO controller 右手

```bash
python scripts/teleop_mujoco_demo.py \
  --mode teleop \
  --xr-source real \
  --xr-repo-root /home/wll/xr_teleoperate \
  --xr-input-mode controller \
  --hand-mode right_only \
  --duration 60 \
  --output-dir data/samples/stage2_right_only_real
```

通过标准：

- 终端显示 `XR source: real xr_teleoperate (...)`
- 终端显示 `websocket is connected`
- 移动右手 controller 时，`xr R`、`raw R -> tgt R` 数值变化
- MuJoCo 中右手目标点和机器人右手同步运动

#### 5.3 真实 PICO 手部追踪右手

如果使用手部追踪而不是 controller：

```bash
python scripts/teleop_mujoco_demo.py \
  --mode teleop \
  --xr-source real \
  --xr-repo-root /home/wll/xr_teleoperate \
  --xr-input-mode hand \
  --hand-mode right_only \
  --duration 60 \
  --output-dir data/samples/stage2_right_only_real
```

#### 5.4 左手单手与双手

右手稳定后，单独检查左手：

```bash
python scripts/teleop_mujoco_demo.py \
  --mode teleop \
  --xr-source real \
  --xr-repo-root /home/wll/xr_teleoperate \
  --xr-input-mode controller \
  --hand-mode left_only \
  --duration 60 \
  --output-dir data/samples/stage2_left_only_real
```

左右手都稳定后，切到双手：

```bash
python scripts/teleop_mujoco_demo.py \
  --mode teleop \
  --xr-source real \
  --xr-repo-root /home/wll/xr_teleoperate \
  --xr-input-mode controller \
  --hand-mode bimanual \
  --duration 60 \
  --output-dir data/samples/stage2_bimanual_real
```

#### 5.5 隐藏 MuJoCo 中的调试球

默认会显示 XR 原始点、目标点和实际末端位置的调试球。录制演示时如需隐藏这些球，添加：

```bash
--no-show-markers
```

示例：

```bash
python scripts/teleop_mujoco_demo.py \
  --mode teleop \
  --xr-source real \
  --xr-repo-root /home/wll/xr_teleoperate \
  --xr-input-mode controller \
  --hand-mode bimanual \
  --duration 60 \
  --output-dir data/samples/stage2_bimanual_real_no_markers \
  --no-show-markers
```

#### 5.6 常见问题

如果报错 `address already in use`：

```text
error while attempting to bind on address ('0.0.0.0', 8012): address already in use
```

说明已有 Vuer 进程占用 `8012`。先检查：

```bash
ss -ltnp | grep 8012
```

关闭旧的 `teleop_hand_and_arm.py` / Vuer 进程后再运行本仓库 demo。

如果网页已连接但红球不动：

- 确认 PICO 里打开的是 `https://vuer.ai?ws=wss://192.168.1.22:8012`
- 确认已点击网页里的 **Virtual Reality**
- 确认允许了 controller / hand tracking 权限
- 移动 controller 时终端里的 `xr R` 应该变化
- 按 trigger / grip 时终端里的 `R trig` / `sqz` 应该变化

如果 PICO 看不到机器人画面，这是正常的。当前 `--display-mode pass-through` 只使用 PICO 采集 XR 输入，MuJoCo 画面看电脑窗口。

### 6. 可视化录制结果

```bash
python scripts/visualize_data.py data/samples/stage1_bimanual_trajectory
```

### 7. 生成目标点样例（工具脚本）

```bash
python scripts/generate_target_points.py --type static --output target_points.npy
python scripts/generate_target_points.py --type multi --output target_points.npy
python scripts/generate_target_points.py --type circle --output target_points.npy
```

## 当前阶段实现说明

- 场景文件：`assets/g1_upper_body_scene.xml`
- 末端控制：`controllers/pd_controller.py`
- 双手控制包装：`controllers/bimanual_controller.py`
- 目标生成：`controllers/target_provider.py`
- 轨迹生成：`controllers/trajectory_generator.py`
- 阶段入口：`scripts/run_stage1.py`
- 阶段 1 统一管线：`scripts/stage1_pipeline.py`
- RGBD 接口：`envs/rgbd_camera.py`
- 阶段 1 日志：`envs/stage1_logger.py`
- bridge：`teleop/bridge.py`
- mock XR 输入：`teleop/mock_xr_input.py`
- 阶段 2 demo：`scripts/teleop_mujoco_demo.py`
- bridge 配置：`configs/bridge.yaml`
- 兼容数据录制：`envs/data_recorder.py`
- 可视化检查：`scripts/visualize_data.py`

当前任务设置：

- 基座固定（weld）
- 非任务关节锁定
- 右手目标 marker：`target_right`（红色）
- 左手目标 marker：`target_left`（蓝色）

阶段 2（任务 2）桥接职责：

- 输入：`head_pose`（可选）、`left_wrist_pose`、`right_wrist_pose`（mock）
- 输出：`left_target_pos`、`right_target_pos`
- 处理链：坐标变换（4x4）-> 缩放（xyz）-> 工作空间裁剪 -> 低通平滑
- 默认原则：bridge 尽量保持输入意图，不在 bridge 内做强限速拖手
- 真实 XR 已支持 `--xr-source real`、`--xr-input-mode controller/hand`、`--hand-mode right_only/left_only/bimanual`
- 可用 `--no-show-markers` 隐藏 MuJoCo viewer 中的目标点和调试球
- 当前不做：抓取/手指控制

## 数据输出格式

每次运行会在 `data/raw/episode_xxx/` 下生成：

```text
episode_xxx/
├── meta.json
├── rgb/
│   ├── 000000.png
│   └── ...
├── depth/
│   ├── 000000.npy
│   └── ...
├── state.npy
├── action.npy
├── target_eef.npy
├── actual_eef.npy
├── timestamps.npy
├── frame_indices.npy
├── camera_intrinsics.npy
├── camera_extrinsics.npy
├── camera_timestamps.npy
└── success.txt
```

`scripts/record_stage1_bimanual_trajectory.py` 的输出格式：

```text
data/samples/stage1_bimanual_trajectory/
├── meta.json
├── rgb/
│   ├── 000000.png
│   └── ...
├── depth/
│   ├── 000000.npy
│   └── ...
├── joint_state.npy
├── action.npy
├── target_eef_left.npy
├── target_eef_right.npy
├── actual_eef_left.npy
├── actual_eef_right.npy
├── tracking_error_left.npy
├── tracking_error_right.npy
├── left_wrist_pose.npy
├── right_wrist_pose.npy
├── raw_target_left.npy
├── raw_target_right.npy
├── transformed_target_left.npy
├── transformed_target_right.npy
├── timestamp.npy
└── camera_meta.json
```

## 阶段 1 最终验收方式

1. 运行可视化跟随：`python scripts/run_stage1.py --duration 10`
2. 录制双手轨迹样本：`python scripts/record_stage1_bimanual_trajectory.py`
3. 检查 `data/samples/stage1_bimanual_trajectory/meta.json` 中 summary 指标：
   - `left.final` / `right.final` 小于 `tracking_threshold`
   - `has_nan=false` 且 `has_divergence=false`
4. 检查 RGBD 输出：
   - `rgb/*.png` 与 `depth/*.npy` 编号对齐
   - `camera_meta.json` 存在
5. 多次重复运行趋势一致（建议至少 3 次）

阶段 1 完成标准：
1. 右手末端能稳定跟随目标点或连续轨迹
2. 左手末端能稳定跟随目标点或连续轨迹
3. 双手同时跟随时无明显发散或 NaN
4. final tracking error left/right 小于配置阈值（默认 `0.05m`）
5. mean tracking error left/right 在合理范围内
6. 能保存 RGB 图像
7. 能保存 depth npy
8. 能保存 joint_state、action、target_eef、actual_eef、tracking_error、timestamp
9. 多次运行结果基本一致

快速检查命令（双手轨迹样本）：

```bash
python - <<'PY'
import os, glob, json, numpy as np
p='data/samples/stage1_bimanual_trajectory'
el=np.load(os.path.join(p,'tracking_error_left.npy'))
er=np.load(os.path.join(p,'tracking_error_right.npy'))
with open(os.path.join(p,'meta.json'),'r',encoding='utf-8') as f:
    meta=json.load(f)
print('steps:',len(el),'rgb:',len(glob.glob(p+'/rgb/*.png')),'depth:',len(glob.glob(p+'/depth/*.npy')))
print('left final/mean:',float(el[-1]),float(np.nanmean(el)))
print('right final/mean:',float(er[-1]),float(np.nanmean(er)))
print('summary:',meta.get('summary',{}))
PY
```

## 说明

- `scripts/run_teleop.sh`、`scripts/train_act.sh`、`scripts/eval_all.sh` 当前仍是占位脚本（后续阶段实现）。
- 若在无显示环境运行，建议设置 `MUJOCO_GL=osmesa`。
- 按指导文档建议，调试顺序优先：右手单手 -> 左手单手 -> 双手同步（便于定位坐标、控制、可达空间问题）。
- 当前脚本已支持 `--hand-mode right_only/left_only/bimanual`，建议先在 `teleop` 下跑稳右手和左手，再切回默认 `bimanual`。
- 使用真实 PICO 时，同一时间只运行一个 Vuer 服务，避免 `8012` 端口冲突。

## 阶段 4 Benchmark 设计骨架

当前已建立 benchmark 的统一接口骨架，但还没有接入真实 ACT / DP3 / GR00T 训练代码。

目录结构：

```text
benchmark/
├── datasets/
│   └── dataset_adapter.py
├── policies/
│   ├── act_runner.py
│   ├── dp3_runner.py
│   └── groot_runner.py
├── evaluators/
│   └── evaluator.py
├── metrics/
│   └── metrics.py
└── run_benchmark.py
```

当前可先对已录制 episode 做数据和指标基准检查：

```bash
python benchmark/run_benchmark.py --data-root data/samples
```

输出：

- 终端打印统一结果表格
- CSV 保存到 `benchmark/results/benchmark_summary.csv`

当前表格中：

- `RecordedDemo` 使用已录制数据中的 tracking error / action 计算指标
- `ACT`、`DP3`、`GR00T` 是接口占位，等待后续接入真实训练和推理代码

第一版 benchmark 数据划分按指导文件默认：

- train: 35
- val: 5
- test: 10
