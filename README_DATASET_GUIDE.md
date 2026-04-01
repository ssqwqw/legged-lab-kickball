# 📚 运动数据集完整指南 - 导航和总结

你提出的两个重要问题已经完全解答！这份文档会帮助你快速找到需要的信息。

## 🎯 你提的两个问题

### Q1: pkl格式的数据集是怎么得到的？

**简答**：Motion Capture → GMR格式 → Legged Lab格式

```
真实人体运动捕捉数据
    ↓
自定义脚本转换 (example_gmr_conversion.py)
    ↓
GMR格式 pickle文件
    ↓
dataset_retarget.py (自动计算key_body_pos)
    ↓
最终 Legged Lab 格式 pickle (walk_and_run/)
    ↓
用于训练的pkl数据集
```

**详细文档**：[PKL_DATA_DOCUMENTATION.md](PKL_DATA_DOCUMENTATION.md)

---

### Q2: 如果我想使用我自己的数据集可以做到吗？

**简答**：完全可以！需要三步转换

```
你的原始数据 (任何格式)
    ↓ 步骤1：转换为GMR格式
你的 GMR pickle
    ↓ 步骤2：运行dataset_retarget.py
你的 Legged Lab pickle
    ↓ 步骤3：配置training config
    ↓
开始训练！
```

**详细文档**：[CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)

---

## 📖 完整文档地图

### 📌 必读（按优先级）

| 文档 | 优先级 | 时长 | 用途 |
|------|--------|------|------|
| **本文件** | ⭐⭐⭐ | 5min | 全局导航 |
| **QUICK_REFERENCE.md** | ⭐⭐⭐ | 5min | 快速查询和命令 |
| **CUSTOM_DATASET_GUIDE.md** | ⭐⭐⭐ | 15min | 如何准备你的数据 |
| **PKL_DATA_DOCUMENTATION.md** | ⭐⭐ | 10min | 理解pkl格式 |
| **OFFICIAL_VS_CUSTOM.md** | ⭐⭐ | 10min | 决策：官方 vs 自己的数据 |
| **FAQ_COMPLETE.md** | ⭐⭐ | 20min | 常见问题解答 |

### 🛠️ 实用脚本（可运行）

| 脚本 | 用途 | 运行方式 |
|------|------|--------|
| **example_gmr_conversion.py** | 演示数据转换 | `bash .codex/run-in-env.sh python example_gmr_conversion.py` |
| **print_pkl_contents.py** | 查看pkl内容 | `bash .codex/run-in-env.sh python print_pkl_contents.py` |
| **analyze_motion_data.py** | 分析运动质量 | `bash .codex/run-in-env.sh python analyze_motion_data.py` |

### 📋 系统文档（参考）

| 组件 | 位置 | 说明 |
|------|------|------|
| **官方pkl数据** | `source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/` | 30个训练用pkl |
| **转换工具** | `scripts/tools/retarget/` | GMR→LabFormat脚本 |
| **DOF配置** | `scripts/tools/retarget/config/g1_29dof.yaml` | 关节映射关系 |
| **训练配置** | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py` | motion_data_dir和weights |

---

## 🚀 五分钟快速开始

### 想用官方数据训练（最快）

```bash
# 1. 直接开始训练
cd /home/user/legged_lab
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless

# ✅ 完成！（官方数据已配置好）
```

**耗时**：0分钟准备 + N天训练

---

### 想用自己的数据训练

```bash
cd /home/user/legged_lab

# 第1步：准备数据（转换为GMR格式）
# 修改 example_gmr_conversion.py 来适配你的数据格式
bash .codex/run-in-env.sh python example_gmr_conversion.py
# 将输出的pkl放在 data/my_gmr_motions/

# 第2步：转换为Legged Lab格式
mkdir -p source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_data

bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 \
    --input_dir data/my_gmr_motions/ \
    --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_data/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --loop clamp

# 第3步：更新训练配置
# 编辑 g1_amp_env_cfg.py:
#   motion_data_dir = ".../data/MotionData/g1_29dof/amp/my_data/"
#   motion_data_weights = {"my_motion_1": 1.0, ...}

# 第4步：开始训练
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless

# ✅ 完成！
```

**耗时**：1-2天准备数据 + N天训练

---

## 🎓 学习路径建议

### 场景1：我是初学者，想快速上手

```
Day 1-2:
  □ 阅读 QUICK_REFERENCE.md
  □ 阅读 OFFICIAL_VS_CUSTOM.md
  □ 用官方数据跑一次训练

Day 3-5:
  □ 阅读 CUSTOM_DATASET_GUIDE.md
  □ 准备你的第一个自定义数据集
  □ 跑一次自定义数据训练
```

### 场景2：我有数据，想快速集成

```
Day 1:
  □ 快速修改 example_gmr_conversion.py
  □ 生成 GMR pickle
  □ 验证使用 print_pkl_contents.py

Day 2:
  □ 运行 dataset_retarget.py
  □ 更新训练配置
  □ 开始训练
```

### 场景3：我想深入理解整个系统

```
Day 1: 阅读所有文档（1-2小时）
  □ 本文件
  □ PKL_DATA_DOCUMENTATION.md
  □ CUSTOM_DATASET_GUIDE.md
  □ FAQ_COMPLETE.md

Day 2-3: 动手实验
  □ 跑 example_gmr_conversion.py（生成合成数据）
  □ 跑 print_pkl_contents.py（查看结构）
  □ 跑 dataset_retarget.py（完整转换）
  □ 查看源代码理解细节

Day 4+：集成你的数据
  □ 编写适配脚本
  □ 训练和优化
```

---

## 💾 核心概念速查

### GMR格式 (输入)

```python
{
    'fps': 30.0,                    # 帧率
    'root_pos': (N, 3) float64,     # 根部位置 (米)
    'root_rot': (N, 4) float32,     # 四元数 (x,y,z,w) ⚠️
    'dof_pos': (N, 29) float64,     # 29个关节角 (弧度)
    'local_body_pos': None,         # 可选
    'link_body_list': None,         # 可选
}
```

### Legged Lab格式 (输出)

```python
{
    'fps': 30.0,                      # 帧率
    'root_pos': (N, 3) float64,       # 根部位置
    'root_rot': (N, 4) float32,       # 四元数 (w,x,y,z) ⚠️ 格式不同！
    'dof_pos': (N, 29) float64,       # 关节角（DOF顺序重新排列）
    'loop_mode': int,                 # 0=CLAMP, 1=WRAP
    'key_body_pos': (N, 6, 3) float32,# 6个关键点位置（自动计算）
}
```

### DOF映射

```
GMR顺序 (MuJoCo):
  [left_hip_pitch, left_hip_roll, left_hip_yaw, ...]

Legged Lab顺序 (Isaac Lab):
  [left_hip_pitch, right_hip_pitch, waist_yaw, ...]
  
转换: 通过 g1_29dof.yaml 中的映射完成
```

---

## 🔧 常用命令速查

### 数据准备

```bash
# 验证官方数据
bash .codex/run-in-env.sh python print_pkl_contents.py

# 分析运动
bash .codex/run-in-env.sh python analyze_motion_data.py

# 转换你的数据（单个文件）
bash .codex/run-in-env.sh python scripts/tools/retarget/single_retarget.py \
    --robot g1 \
    --input_file data/my.pkl \
    --output_file data/out.pkl \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml

# 批量转换
bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 \
    --input_dir data/my_gmr/ \
    --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_data/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml
```

### 训练

```bash
# 用官方数据
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless

# 用自己的数据（需要先更新motion_data_dir）
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless
```

### 调试

```bash
# 查看环境变量
echo $CODEX_CONDA_ENV
echo $ISAACLAB_PATH

# 测试Python环境
bash .codex/run-in-env.sh python -c "import isaaclab; print(isaaclab.__version__)"
```

---

## ⚠️ 常见陷阱

| 陷阱 | 症状 | 修复 |
|------|------|------|
| **四元数格式错** | "nan值" 或 "奇怪的旋转" | 确保是(x,y,z,w)不是(w,x,y,z) |
| **DOF顺序错** | "关节反向运动" 或 "跌倒" | 检查config中gmr_dof_names顺序 |
| **高度不合理** | "机器人在空中" 或 "穿过地面" | 检查root_pos[:,2]范围(0.6-1.0m) |
| **数据抖动** | "关节震颤" | 使用滤波或重新采集 |
| **文件找不到** | "No motion files found" | 检查motion_data_dir路径和文件名 |

---

## 📞 快速问题解答

### Q: 哪个文档我必须读？

A: 从这个顺序开始：
1. **本文件** (5分钟) - 全局理解
2. **QUICK_REFERENCE.md** (5分钟) - 命令查询
3. **CUSTOM_DATASET_GUIDE.md** (15分钟) - 如果你有数据

### Q: 我没有自己的数据，能训练吗？

A: 完全可以！有30个官方数据可直接使用。参考 **OFFICIAL_VS_CUSTOM.md**。

### Q: 转换需要多长时间？

A: 
- 单个文件：~20秒（含可视化）
- 30个文件：~2分钟（批量）

### Q: 我的数据格式特殊怎么办？

A: 参考 **CUSTOM_DATASET_GUIDE.md** 中的多个示例，改 `example_gmr_conversion.py` 适配。

### Q: 还有问题？

A: 查 **FAQ_COMPLETE.md**，有28个常见问题的详细回答。

---

## 🎯 下一步行动

### ➡️ 立即开始（选一个）：

#### 选项A：用官方数据训练（立即）
```bash
cd /home/user/legged_lab
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg --headless
```

#### 选项B：理解官方数据格式
```bash
# 阅读
cat PKL_DATA_DOCUMENTATION.md

# 运行
bash .codex/run-in-env.sh python print_pkl_contents.py
bash .codex/run-in-env.sh python analyze_motion_data.py
```

#### 选项C：准备你的数据
```bash
# 阅读
cat CUSTOM_DATASET_GUIDE.md

# 修改脚本
nano example_gmr_conversion.py

# 运行
bash .codex/run-in-env.sh python example_gmr_conversion.py
```

---

## 📊 一览表：官方数据 vs 自己的数据

| 方面 | 官方数据 | 自己的数据 |
|------|---------|----------|
| **开箱即用** | ✅ 立即可训练 | ❌ 需要转换 |
| **灵活性** | ❌ 固定30个动作 | ✅ 完全自定义 |
| **需要准备** | ❌ 无 | ✅ 1-2天 |
| **数据量** | 📊 30个动作 | 📊 可自由选择 |
| **质量** | ✅ 高质量验证 | 🟡 取决于源数据 |
| **适合场景** | 学习、baseline | 应用、自定义 |

---

## ✅ 完成检查列表

如果想完成"使用自己的数据"的完整流程：

- [ ] 阅读本导航文件
- [ ] 阅读 CUSTOM_DATASET_GUIDE.md
- [ ] 准备你的原始数据
- [ ] 修改 example_gmr_conversion.py
- [ ] 运行转换脚本生成GMR pkl
- [ ] 验证数据（print_pkl_contents.py）
- [ ] 运行 dataset_retarget.py
- [ ] 更新 g1_amp_env_cfg.py
- [ ] 启动训练
- [ ] 调整超参数和数据

---

## 📝 最后的话

我已为你准备了完整的工具和文档体系：

✅ **理论**：6份详细文档  
✅ **实践**：3个可运行的脚本  
✅ **参考**：命令速查和FAQ  

**总结答案**：

1️⃣ **pkl数据怎么来的**？  
   Motion Capture → GMR → (dataset_retarget.py) → Legged Lab pkl

2️⃣ **能用自己的数据吗**？  
   完全可以！三步转换：自定义数据 → GMR → Legged Lab → 训练

---

## 🎓 推荐阅读顺序

```
你在这里 (导航)
  ↓
QUICK_REFERENCE.md (5分钟快速查询)
  ↓
根据你的需求选择：
  ├─ 用官方数据 → OFFICIAL_VS_CUSTOM.md + 开始训练
  └─ 用自己数据 → CUSTOM_DATASET_GUIDE.md + example_gmr_conversion.py
```

祝你使用愉快！有问题查 **FAQ_COMPLETE.md** 或相关文档。🚀

---

**最后更新**：2026年4月1日  
**版本**：完整版 v1.0  
**包含**：官方数据分析 + 自定义数据指南 + 工具脚本
