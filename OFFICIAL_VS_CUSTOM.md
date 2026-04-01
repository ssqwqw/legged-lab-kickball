# 官方数据 vs 自己的数据使用对比

## 📊 两种方式的对比

| 方面 | 使用官方数据 | 使用自己的数据 |
|------|------------|-------------|
| **难度** | ✅ 简单，开箱即用 | 🟡 中等，需要转换 |
| **时间** | ✅ 立即可训练 | 🟡 需要预处理 |
| **灵活性** | ❌ 固定的动作集 | ✅ 完全自定义 |
| **质量** | ✅ 高质量运动 | 🟡 取决于你的数据 |
| **数据源** | 📍 已有30个pkl | 📍 你的Motion Capture/算法生成 |
| **自定义性** | ❌ 受限 | ✅ 无限制 |

## 🎯 什么时候用官方数据，什么时候用自己的

### 使用官方数据的场景 ✅

```python
# 1. 快速开始、学习框架
from legged_lab.tasks.locomotion.amp.config.g1 import G1AMPEnvCfg

# 2. 验证训练流程是否正确
# 3. 与论文结果对比
# 4. 作为baseline
```

### 使用自己数据的场景 ✅

```
1. 你有真实运动捕捉数据（Motion Capture）
2. 你想加入新的运动类型（如跳跃、爬行等）
3. 你要对接其他运动生成算法的输出
4. 你想微调运动以适应你的场景
5. 你需要特定的动作组合
```

## 🔄 我该如何选择？

```
                    ┌─── 你有运动数据吗？
                    │
        ┌───YES─────┴─────NO───┐
        │                       │
        ▼                       ▼
    你想快速开始？         使用官方数据
        │                   (30个pkl)
    ┌───┴──────┐
    │          │
   YES        NO
    │          │
    ▼          ▼
 使用官方      使用自己
  数据        的数据
            ├─ 转换为GMR
            ├─ 运行dataset_retarget.py
            ├─ 更新配置
            └─ 开始训练
```

## 📋 数据流对比

### 官方数据流

```
官方pkl文件 (walk_and_run/) 
    ↓ (已是Legged Lab格式)
在训练中直接使用
    ↓
MotionDataManager载入
    ↓
训练！
```

### 自己数据流

```
你的原始数据 (CSV/BVH/H5等)
    ↓
转换为GMR (example_gmr_conversion.py)
    ↓
GMR pkl文件
    ↓
运行dataset_retarget.py
    ↓
Legged Lab pkl文件
    ↓
在训练中直接使用
    ↓
MotionDataManager载入
    ↓
训练！
```

## 📌 实际例子

### 例1：使用官方数据训练

```bash
# 直接开始训练，无需任何准备
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless

# 配置中已指向官方数据：
# motion_data_dir = ".../data/MotionData/g1_29dof/amp/walk_and_run"
```

### 例2：使用自己的运动捕捉数据

```bash
# 第1天：准备数据
# 有3D人体动作捕捉数据 (BVH格式)

# 第2天：转换数据
python3 << 'EOF'
import bvh
import numpy as np
from example_gmr_conversion import create_gmr_data_from_arrays, validate_gmr_data, save_gmr_data

# 加载BVH
with open('my_mocap.bvh') as f:
    mocap = bvh.parse(f)

# 提取运动数据（需要你的BVH格式知识）
root_pos = ...
root_euler = ...
dof_pos = ...

# 转换
gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos)
gmr_data = validate_gmr_data(gmr_data)
save_gmr_data(gmr_data, 'my_motion.pkl')
EOF

# 第3天：转换为Legged Lab格式
mkdir -p data/my_gmr_motions
# 把所有GMR pkl放在这个目录

bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 \
    --input_dir data/my_gmr_motions/ \
    --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_mocap_data/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml

# 第4天：修改训练配置
# 编辑 g1_amp_env_cfg.py
# motion_data_dir = ".../data/MotionData/g1_29dof/amp/my_mocap_data"

# 第5天：开始训练
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless
```

## 📊 数据质量对比

### 官方数据
```
✅ 优势：
   - 经过验证，质量有保证
   - 30个动作，覆盖走、跑、转身、侧步等
   - 帧率统一（~30 FPS）
   - 已经过运动学检查
   - 适合初学者参考

❌ 劣势：
   - 动作集有限
   - 来自特定运动捕捉数据
   - 可能不适合你的特定场景
```

### 自己的数据
```
✅ 优势：
   - 可以包含特定的动作
   - 可以添加新的运动类型
   - 可以对接你的现有系统
   - 100%可定制

❌ 劣势：
   - 需要检查质量
   - 可能有抖动、不稳定
   - 需要满足数据格式要求
   - 需要额外的预处理时间
```

## 🚦 决策树：我应该先做什么？

```
你是第一次使用这个框架吗？
        │
    ┌───┴──────┐
   YES        NO
    │          │
    ▼          │
使用官方数据    你有自己的数据吗？
学习框架        │
│          ┌───┴───┐
│         YES     NO
│          │       │
│          ▼       ▼
│       准备      继续
│       数据     使用官方
│       &        数据或
│      转换      找数据来源
│
└─ 熟悉后 ──► 尝试自己的数据
   (1-2天)
```

## 🎓 学习路径建议

### Week 1：熟悉框架
```
Day 1-2: 使用官方数据训练一个基础的G1 AMP模型
Day 3-5: 理解motion_data.py和MotionDataManager如何工作
```

### Week 2：准备自己的数据
```
Day 1-2: 收集你的运动数据（Motion Capture或其他来源）
Day 3-4: 转换为GMR格式并验证
Day 5: 运行dataset_retarget.py
```

### Week 3：集成和培训
```
Day 1: 更新配置文件
Day 2-5: 用自己的数据训练并调优
```

## 💡 技巧：混合官方和自己的数据

你也可以同时使用两者！

```python
# 在g1_amp_env_cfg.py中

class MotionDataCfg:
    # 指向包含所有pkl的目录
    motion_data_dir = os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "data", "MotionData", "g1_29dof", "amp", "mixed_data"
    )
    
    # 混合权重：官方数据 + 你的数据
    motion_data_weights = {
        # 官方数据
        "B10_-__Walk_turn_left_45_stageii": 1.0,
        "B15_-__Walk_turn_around_stageii": 1.0,
        "C3_-_run_stageii": 1.0,
        
        # 你的数据
        "my_custom_jump": 2.0,        # 跳跃（权重高，更常用）
        "my_custom_side_flip": 1.0,   # 翻滚
        "my_custom_dance": 0.5,       # 舞蹈（权重低）
    }
```

## 📞 如果出问题怎么办？

| 问题 | 解决方案 |
|------|--------|
| GMR转换失败 | ✅ 检查数据形状和类型 |
| dataset_retarget崩溃 | ✅ 查看关节角度范围 |
| 训练不收敛 | ✅ 检查运动是否抖动/不稳定 |
| 模拟器行为奇怪 | ✅ 验证root_pos高度(0.6-1.0m) |
| DOF映射错误 | ✅ 检查配置文件中DOF顺序 |

## 🎯 总结

**简答题：我能用自己的数据吗？**
✅ 是的，完全可以！遵循这个流程：
1. 准备数据（GMR格式）
2. 运行dataset_retarget.py
3. 更新训练配置
4. 开始训练！

预计准备时间：**1-2天**（数据转换）+ **1-5天**（训练）
