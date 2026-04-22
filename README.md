# mujoco_vla_project

## 项目简介

本项目用于完成以下链路：

- 在 MuJoCo 中搭建 G1 上半身任务场景
- 实现上半身末端目标位置跟随（当前为右手）
- 打通 RGBD 相机读取与数据录制
- 为后续 XR 遥操接入、数据采集与 benchmark 做准备

当前代码状态：**阶段 1 核心闭环可运行**（上半身末端跟随 + RGBD 接口 + 同步存储）。

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

## 快速开始（阶段 1）

### 1. 运行末端跟随 + RGBD 采集

```bash
conda activate mujoco_vla
python scripts/run_stage1.py --duration 10 --save-video
```

或使用脚本入口：

```bash
bash scripts/record_data.sh 10 data/raw
```

### 2. 可视化最新 episode

```bash
python scripts/visualize_data.py --latest
```

### 3. 生成目标点样例（多点/轨迹）

```bash
python scripts/generate_target_points.py --type static --output target_points.npy
python scripts/generate_target_points.py --type multi --output target_points.npy
python scripts/generate_target_points.py --type circle --output target_points.npy
```

## 当前阶段实现说明

- 场景文件：`assets/g1_upper_body_scene.xml`
- 末端控制：`controllers/pd_controller.py`
- 阶段入口：`scripts/run_stage1.py`
- RGBD 接口：`envs/rgbd_camera.py`
- 数据录制：`envs/data_recorder.py`
- 可视化检查：`scripts/visualize_data.py`

当前任务设置：

- 基座固定（weld）
- 非任务关节锁定
- 右手末端跟随红色目标点 `target_right`

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

## 阶段 1 验收建议

1. 能加载场景并看到右手追踪红球  
2. 误差曲线下降并稳定在小范围  
3. `rgb/depth/state/target/actual/timestamp` 帧数一致  
4. 多次运行结果趋势一致

快速检查命令（最新 episode）：

```bash
python - <<'PY'
import os, glob, numpy as np
base='data/raw'
ep=sorted([d for d in os.listdir(base) if d.startswith('episode_')])[-1]
p=os.path.join(base,ep)
state=np.load(os.path.join(p,'state.npy'))
target=np.load(os.path.join(p,'target_eef.npy'))
actual=np.load(os.path.join(p,'actual_eef.npy'))
ts=np.load(os.path.join(p,'timestamps.npy'))
cts=np.load(os.path.join(p,'camera_timestamps.npy'))
err=np.linalg.norm(actual-target,axis=1)
print('episode:',ep)
print('steps:',len(ts),'rgb:',len(glob.glob(p+'/rgb/*.png')),'depth:',len(glob.glob(p+'/depth/*.npy')))
print('final_error:',float(err[-1]),'mean_error:',float(err.mean()))
print('max|camera_ts-ts|:',float(np.abs(cts-ts).max()))
PY
```

## 说明

- `scripts/run_teleop.sh`、`scripts/train_act.sh`、`scripts/eval_all.sh` 当前仍是占位脚本（后续阶段实现）。
- 若在无显示环境运行，建议设置 `MUJOCO_GL=osmesa`。
