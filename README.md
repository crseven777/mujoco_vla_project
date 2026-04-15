# mujoco_vla_project


## 项目简介

本项目旨在完成以下目标：

* 在 MuJoCo 中搭建 Unitree G1 上半身仿真场景
* 实现上半身末端（双手）目标位置跟随控制
* 接入 XR 遥操输入，实现 XR → MuJoCo 闭环
* 采集标准化上半身遥操数据集
* 在统一接口下 Benchmark ACT / GR00T / DP3 等策略

当前阶段：**阶段 0（环境配置与项目初始化）**

---

## 一、环境要求

### 硬件要求

* Ubuntu 22.04（推荐）
* NVIDIA GPU（推荐 RTX 3060 及以上）
* 至少 16GB 内存

### 软件要求

* Miniconda / Anaconda
* Git
* Python 3.10
* MuJoCo 3.x

---

## 二、环境安装步骤

## 1. 创建 Conda 环境

```bash
conda create -n mujoco_vla python=3.10 -y
conda activate mujoco_vla
```

---

## 2. 安装系统依赖（Ubuntu）

```bash
sudo apt update
sudo apt install -y \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    libxrender1 \
    libxext6 \
    libx11-6 \
    patchelf \
    tree
```

---

## 3. 安装 Python 依赖

```bash
pip install --upgrade pip
pip install mujoco glfw numpy scipy matplotlib opencv-python imageio tqdm
```

---

## 4. 验证 MuJoCo 安装

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

若成功输出版本号，说明安装完成。

---

## 三、项目目录结构

```text
mujoco_vla_project/
├── README.md
├── test_mujoco.py                  # Mujoco 基础测试脚本
├── assets/                         # Robot XML / URDF / 场景资源
├── configs/                        # 相机 / 控制器参数
├── teleop/                         # XR 遥操桥接模块
├── envs/                           # Mujoco 环境封装
├── controllers/                    # IK / tracking controller
├── data/
│   ├── raw/                        # 原始遥操录制数据
│   ├── processed/                  # 处理后的训练数据
│   └── samples/                    # 示例 episode
├── benchmark/
│   ├── datasets/                   # 数据适配器
│   ├── policies/                   # ACT / DP3 / GR00T
│   ├── evaluators/                 # 测试评估模块
│   └── results/                    # benchmark 输出结果
├── scripts/
│   ├── launch_scene.sh             # 启动 Mujoco 场景
│   ├── run_teleop.sh               # 启动 XR 遥操
│   ├── record_data.sh              # 数据录制
│   └── run_benchmark.sh            # benchmark 总入口
└── docs/
    └── undergrad_guide.md
```

---

## 四、如何启动 MuJoCo 场景

## 1. 基础测试场景

```bash
python test_mujoco.py
```

运行后应看到：

* 地面平面
* 一个方块自由下落

说明 MuJoCo Viewer 正常。

---

## 2. 启动 G1 上半身场景（后续阶段）

```bash
bash scripts/launch_scene.sh
```

说明：

该脚本负责：

* 加载 G1 上半身 XML 模型
* 初始化相机
* 启动控制循环

---

## 五、如何启动 XR 遥操模块

XR Teleoperate 仓库建议单独管理：

```text
~/xr_teleoperate/
```

### 1. 克隆 XR 仓库

```bash
git clone --depth=1 https://github.com/unitreerobotics/xr_teleoperate.git
```

---

### 2. 启动 XR 遥操（后续阶段）

```bash
bash scripts/run_teleop.sh
```

说明：

该脚本负责：

* 启动 XR 输入读取
* 坐标系转换
* 输出末端目标位置
* 与 MuJoCo 通信

---

## 六、数据保存位置

所有录制数据统一保存在：

```text
data/raw/
```

推荐单条 episode 格式：

```text
episode_001/
├── meta.json
├── rgb/
├── depth/
├── state.npy
├── action.npy
├── target_eef.npy
├── actual_eef.npy
└── success.txt
```

说明：

* rgb：相机彩色图像
* depth：深度图
* state：机器人状态
* action：控制输入
* target_eef：目标末端轨迹
* actual_eef：实际末端轨迹

---

## 七、Benchmark 入口脚本

统一 benchmark 启动入口：

```bash
bash scripts/run_benchmark.sh
```

后续功能：

* 加载 train / val / test 数据
* 启动 ACT / DP3 / GR00T
* 输出 success rate / tracking error / latency 等指标

结果保存在：

```text
benchmark/results/
```

---

## 八、当前阶段已完成内容（阶段 0）

请按实际进度勾选：

* [ ] Conda 环境创建完成
* [ ] MuJoCo 安装完成
* [ ] MuJoCo demo 可运行
* [ ] 项目目录创建完成
* [ ] README 完成
* [ ] XR 仓库下载完成

---

## 九、快速复现指南

新用户拿到仓库后：

```bash
git clone <your_repo>
cd mujoco_vla_project

conda create -n mujoco_vla python=3.10 -y
conda activate mujoco_vla

pip install mujoco glfw numpy scipy matplotlib opencv-python imageio tqdm

python test_mujoco.py
```

若看到 MuJoCo Viewer 正常弹出，即说明环境配置成功。

---

## 十、后续计划

* 阶段 1：G1 上半身末端跟随 + RGBD 接口
* 阶段 2：XR 遥操接入
* 阶段 3：采集 50 条标准化上半身数据
* 阶段 4：统一 Benchmark ACT / GR00T / DP3
