# mujoco_vla_project

## 项目简介

本项目用于完成以下链路：

- 在 MuJoCo 中搭建 G1 上半身任务场景
- 实现上半身末端目标位置跟随（当前为右手）
- 打通 RGBD 相机读取与数据录制
- 为后续 XR 遥操接入、数据采集与 benchmark 做准备

当前代码状态：

- **阶段 1 已完成**：双手末端连续轨迹跟随 + RGBD + 同步存储
- **阶段 2 任务 2 已实现**：`mock XR -> bridge -> Mujoco target`（不含真实 XR 设备接入）

## 环境要求

- Ubuntu 22.04（推荐）
- Python 3.10
- MuJoCo 3.x
- Conda（推荐）

## 安装步骤

```bash
conda create -n mujoco_vla python=3.10 -y
conda activate mujoco_vla

pip install --upgrade pip
pip install mujoco glfw numpy scipy matplotlib opencv-python imageio tqdm
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

`trajectory` 模式（沿用阶段 1 目标生成）：

```bash
python scripts/teleop_mujoco_demo.py --mode trajectory
```

调快手臂响应（可选）：

```bash
python scripts/teleop_mujoco_demo.py --mode trajectory --max-joint-speed 1.5 --task-gain 3.5
```

### 5. 可视化录制结果

```bash
python scripts/visualize_data.py data/samples/stage1_bimanual_trajectory
```

### 6. 生成目标点样例（工具脚本）

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
- 当前不做：真实 XR 设备接入、网络通信、抓取/手指控制

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
