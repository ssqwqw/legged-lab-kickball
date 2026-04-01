# 🤖 Legged Lab（中文说明）

[English](README.md) | 简体中文

基于 Isaac Lab 的腿式机器人强化学习扩展仓库，当前重点支持 Unitree G1 的 DeepMimic 与 AMP。

## 功能概览

- DeepMimic（G1）
- AMP（Adversarial Motion Priors，G1）
- 动作数据转换与回放工具

## 环境要求

- Isaac Lab 2.3.x
- Isaac Sim 5.1.x
- Python 3.11
- Git LFS

## 安装

```bash
git clone https://github.com/ssqwqw/legged-lab-kickball.git
cd legged-lab-kickball
git lfs install
git lfs pull
```

安装主包：

```bash
python -m pip install -e source/legged_lab
```

## 重要：命令运行方式

本仓库内所有 Python 命令建议通过环境包装器执行：

```bash
bash .codex/run-in-env.sh python <script>.py [args]
```

例如训练：

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
```

## 训练与回放

### DeepMimic 训练

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
```

### DeepMimic 回放

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/play.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --num_envs 1 --checkpoint logs/rsl_rl/<exp>/<run>/model_xxx.pt
```

### AMP 训练

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000
```

### AMP 回放

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 1 --checkpoint logs/rsl_rl/<exp>/<run>/model_xxx.pt
```

## 数据与产物

- 动作数据：`source/legged_lab/legged_lab/data/MotionData`
- 训练日志：`logs/`
- 运行输出：`outputs/`
- 临时文件：`temp/`

## 不上传训练权重

仓库已配置忽略常见训练权重与导出产物（如 `model_*.pt`、`exported/` 等）。

建议仅提交：

- 源码
- 配置
- 文档

不提交：

- 训练日志
- 模型权重
- 导出模型文件
