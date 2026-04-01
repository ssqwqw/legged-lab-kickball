# 常见问题解答 (FAQ)

## 📚 官方数据相关

### Q1: 官方数据（walk_and_run）是从哪来的？

**A:** 官方数据来源流程：
1. **原始数据**：通常来自Motion Capture（真人运动捕捉）
2. **GMR格式**：转换为中间格式（包含fps, root_pos, root_rot, dof_pos）
3. **Legged Lab格式**：通过 `dataset_retarget.py` 转换
4. **最终位置**：`source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/`

所以这些pkl文件**已经是可以直接使用的最终格式**。

### Q2: 官方的30个pkl文件包含什么？

**A:** 共30个动作：
- **行走**：16个（包括转身、侧步、后退）
- **跑步**：14个（包括不同方向的转身）
- **总时长**：约156秒（2.6分钟）
- **每个文件包含**：
  - fps（帧率，~30）
  - root_pos、root_rot、dof_pos（完整运动轨迹）
  - loop_mode（CLAMP，表示到末尾停止而不是循环）
  - key_body_pos（6个关键点的位置，用于奖励计算）

### Q3: 为什么需要key_body_pos？

**A:** 用于AMP训练中的：
- 动作匹配奖励（让学习的动作与示范动作相似）
- 接触检测（脚何时着地）
- 运动质量评估

自动计算！在转换过程中，`dataset_retarget.py` 会通过模拟器自动计算所有关键点位置。

---

## 🎯 使用自己数据的问题

### Q4: 我能用自己的数据吗？

**A:** ✅ **完全可以！** 

三个步骤：
1. 转换为GMR格式（pickle字典）
2. 运行 `dataset_retarget.py`
3. 更新训练配置

所有工具都已提供。参考：[CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)

### Q5: 我需要30个动作吗？

**A:** 不需要！推荐量：
- **最少**：3-5个基本动作（走、跑、转身）
- **不错**：10-15个多样化动作
- **很好**：30个+（官方规模）

5个高质量动作比30个低质量动作更有效。

### Q6: 我的数据是CSV/JSON格式，怎么办？

**A:** 使用 `example_gmr_conversion.py` 中的函数：

```python
import numpy as np
from example_gmr_conversion import (
    create_gmr_data_from_arrays, 
    validate_gmr_data, 
    save_gmr_data
)

# 1. 加载你的数据（任何格式）
root_pos = np.array(...)    # shape: (N, 3)
root_euler = np.array(...)  # shape: (N, 3)
dof_pos = np.array(...)     # shape: (N, 29)

# 2. 转换为GMR
gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos)

# 3. 验证
gmr_data = validate_gmr_data(gmr_data)

# 4. 保存
save_gmr_data(gmr_data, 'my_motion.pkl')
```

### Q7: 我有BVH文件怎么处理？

**A:** BVH是Motion Capture标准格式。步骤：

```bash
# 1. 安装BVH库
bash .codex/run-in-env.sh pip install bvh

# 2. 编写加载脚本
```

```python
import bvh

with open('motion.bvh') as f:
    mocap_data = bvh.parse(f)

# 3. 提取root_pos（由你的BVH结构决定）
# 4. 提取关节角度（跳过根部6DOF）
# 5. 使用create_gmr_data_from_arrays转换
```

参考：[CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)中的BVH部分

### Q8: 我的四元数格式是(w,x,y,z)，而不是(x,y,z,w)？

**A:** 需要重新排列：

```python
quat_wxyz = np.array([...])  # 现有的 (w,x,y,z)
quat_xyzw = quat_wxyz[:, [1,2,3,0]]  # ✅ 转换为 (x,y,z,w)

# 或者如果你有scipy
from scipy.spatial.transform import Rotation as R
quat_xyzw = R.from_quat(quat_wxyz[:, [1,2,3,0]]).as_quat()
```

⚠️ **这是常见的陷阱，一定要检查！**

### Q9: 如何从欧拉角生成四元数？

**A:** 使用scipy：

```python
from scipy.spatial.transform import Rotation as R

euler_angles = np.array([...])  # shape: (N, 3)，单位：弧度
# 假设顺序是 (roll, pitch, yaw)，即 XYZ

quaternions = R.from_euler('xyz', euler_angles).as_quat()
# ✅ 返回 (x, y, z, w) 格式

# 注意：如果欧拉角顺序不同，改变 'xyz'
# 例如：'zyx' 表示 (yaw, pitch, roll)
```

### Q10: 我需要计算key_body_pos吗？

**A:** 不需要！`dataset_retarget.py` 会自动计算。

你的GMR数据只需要提供：
- fps ✅
- root_pos ✅
- root_rot ✅
- dof_pos ✅

`key_body_pos` 会自动生成。

---

## ⚙️ 技术细节

### Q11: GMR和Legged Lab格式有什么区别？

**A:** 主要是DOF顺序不同：

```
GMR格式（MuJoCo顺序）：
  [left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, ...]

Legged Lab格式（Isaac Lab顺序）：
  [left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...]
```

`dataset_retarget.py` 中有DOF映射逻辑，通过配置文件(`g1_29dof.yaml`)完成转换。

### Q12: 什么是loop_mode？

**A:** 定义动作到末尾时的行为：

```
loop_mode = 0 (CLAMP)：
  动作播放完毕，停留在最后一帧
  使用场景：单一动作（不循环）

loop_mode = 1 (WRAP)：
  动作循环播放，末帧→首帧
  使用场景：周期性运动（走路、跑步）
```

官方数据全部用CLAMP。

### Q13: 为什么需要fps字段？

**A:** 用于计算速度和角速度：

```python
dt = 1.0 / fps  # 时间步长

# 在MotionDataManager中：
root_vel = (root_pos[t+1] - root_pos[t]) / dt  # 线性速度
ang_vel = angular_velocity_from_quat_diff(root_rot, dt)  # 角速度
```

不同动作可以有不同fps（29-30 FPS都是正常的）。

---

## 🛠️ 转换工具相关

### Q14: single_retarget.py 和 dataset_retarget.py 的区别？

**A:** 

| 工具 | 用途 | 输入 | 输出 |
|------|------|------|------|
| **single_retarget.py** | 单个文件 + 可视化 | 1个GMR pkl | 1个Legged Lab pkl |
| **dataset_retarget.py** | 批量转换 + 高效 | 整个目录的GMR pkl | 整个目录的Legged Lab pkl |

**选择建议**：
- 调试/学习：用 `single_retarget.py`（可以看可视化）
- 生产/批量：用 `dataset_retarget.py`（快速）

### Q15: 如何只转换特定的帧范围？

**A:** 使用 `single_retarget.py` 的 `--frame_range` 参数：

```bash
bash .codex/run-in-env.sh python scripts/tools/retarget/single_retarget.py \
    --robot g1 \
    --input_file my_motion.pkl \
    --output_file my_motion_clip.pkl \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --frame_range 10 100  # ✅ 只转换第10-100帧
```

### Q16: 转换过程很慢吗？

**A:** 
- **single_retarget.py**：~10-30秒/个文件（包括可视化）
- **dataset_retarget.py**：~1-2分钟/30个文件（批量高效）

主要耗时：模拟器计算key_body_pos。

---

## 📊 数据验证

### Q17: 如何检查我的数据质量？

**A:** 运行验证脚本：

```python
from example_gmr_conversion import validate_gmr_data

gmr_data = pickle.load(open('my_motion.pkl', 'rb'))
gmr_data = validate_gmr_data(gmr_data)
```

检查项：
- ✅ 形状是否正确
- ✅ 数据类型是否正确
- ✅ 四元数是否规范化
- ✅ 值范围是否合理
- ⚠️ 高度是否在0.6-1.0m范围

### Q18: 关节角度超出范围怎么办？

**A:** 可能的原因和解决方案：

```
范围问题 → 原因 → 解决
-1.5~1.5弧度太小 → 某些关节有更大范围 → 检查物理模型
-86°~86°太窄 → 这是正常范围 → 数据可能有问题

如果确实需要更大范围，编辑机器人URDF文件改joint limits
```

### Q19: 数据中有抖动或跳跃怎么办？

**A:** 这表示运动数据质量问题。解决方案：

```python
# 1. 低通滤波
from scipy.ndimage import uniform_filter1d

smoothed_pos = uniform_filter1d(root_pos, size=5, axis=0, mode='nearest')

# 2. 样条插值
from scipy.interpolate import UnivariateSpline

# 3. 降采样
# 如果太冗余，可以隔帧采样

# 重新保存为GMR
gmr_data = create_gmr_data_from_arrays(
    smoothed_pos, root_euler, dof_pos
)
```

---

## 🚀 训练相关

### Q20: 如何在训练中使用自己的数据？

**A:** 三步：

```python
# 1. 编辑 g1_amp_env_cfg.py

class MotionDataCfg:
    motion_data_dir = os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "data/MotionData/g1_29dof/amp/my_custom_data"  # ← 改这里
    )
    
    motion_data_weights = {
        "my_motion_1": 1.0,
        "my_motion_2": 1.0,
        # ... 所有你转换的动作
    }

# 2. 开始训练
# bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
#     --task g1_amp_env_cfg --headless

# 3. 模型会自动从motion_data_weights中随机采样动作进行训练
```

### Q21: 动作权重什么意思？

**A:** 采样概率。例如：

```python
motion_data_weights = {
    "walk": 2.0,      # 比例: 2/(2+1+1) = 50%
    "run": 1.0,       # 比例: 1/(2+1+1) = 25%
    "turn": 1.0,      # 比例: 1/(2+1+1) = 25%
}
```

权重高的动作在训练中被使用更频繁。

### Q22: 用自己的数据训练效果不好？

**A:** 检查清单：

- [ ] 数据质量？（抖动/不稳定）
- [ ] 数据量？（太少会过拟合）
- [ ] 多样性？（缺乏变化）
- [ ] 配置正确？（motion_data_dir指向正确位置）
- [ ] DOF映射？（关节顺序是否正确）
- [ ] 超参数？（学习率、网络大小等）

**建议**：先用官方数据验证训练流程工作正常，再替换为自己的数据。

---

## 📁 文件位置速查

### Q23: 关键文件都在哪？

**A:**

```
转换工具：
  scripts/tools/retarget/
    ├── gmr_to_lab.py           ← 核心实现
    ├── single_retarget.py      ← 单文件
    ├── dataset_retarget.py     ← 批量
    └── config/g1_29dof.yaml    ← DOF映射

数据目录：
  source/legged_lab/legged_lab/data/MotionData/
    └── g1_29dof/amp/
        ├── walk_and_run/        ← 官方数据
        └── [你的数据]/          ← 放这里

训练配置：
  source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/
    └── g1_amp_env_cfg.py       ← 修改motion_data_dir

辅助脚本（我创建的）：
  ├── example_gmr_conversion.py ← 数据转换示例
  ├── print_pkl_contents.py     ← 检查pkl内容
  ├── analyze_motion_data.py    ← 分析运动
  └── 各种文档：
      ├── CUSTOM_DATASET_GUIDE.md
      ├── QUICK_REFERENCE.md
      ├── OFFICIAL_VS_CUSTOM.md
      └── 这个FAQ
```

### Q24: 为什么我的文件找不到？

**A:** 常见问题：

```
错误信息\原因\解决方案

"No .pkl files found"→ 文件不在指定目录 → 检查--input_dir路径

"Motion name X not found"→ motion_data_weights中的名称与文件不匹配 → 检查文件名和weights键是否一致

"DOF name X not found"→ 配置文件中DOF名称错误 → 使用标准配置或检查拼写
```

---

## 🎓 学习资源

### Q25: 我想更深入了解，有文档吗？

**A:** 我为你准备了完整的文档集：

| 文档 | 用途 |
|------|------|
| **QUICK_REFERENCE.md** | ⭐ 快速查询，包含所有命令 |
| **CUSTOM_DATASET_GUIDE.md** | 详细的数据准备指南 |
| **OFFICIAL_VS_CUSTOM.md** | 官方数据 vs 自己的数据 |
| **PKL_DATA_DOCUMENTATION.md** | PKL格式详解 |
| **这个FAQ** | 常见问题回答 |

**建议阅读顺序**：
1. QUICK_REFERENCE.md（全局理解）
2. CUSTOM_DATASET_GUIDE.md（深入细节）
3. 各个脚本的代码注释

### Q26: 有没有完整的例子我可以跑？

**A:** ✅ 有！三个脚本都是可运行的：

```bash
# 1. 生成合成数据并验证
bash .codex/run-in-env.sh python example_gmr_conversion.py

# 2. 查看pkl文件结构
bash .codex/run-in-env.sh python print_pkl_contents.py

# 3. 分析运动数据质量
bash .codex/run-in-env.sh python analyze_motion_data.py
```

输出会展示如何处理数据。

---

## 🆘 故障排除

### Q27: 运行dataset_retarget.py时崩溃

**A:** 检查：

```
错误信息 → 可能原因 → 解决方案

CUDA内存不足 → GPU显存不够 → 用--headless，或减少--num_envs

关节超限 → DOF值大过物理范围 → 检查数据范围或DOF映射

四元数无效 → 四元数未规范化或格式错 → 检查root_rot是否(x,y,z,w)

模型加载失败 → 机器人配置文件问题 → 检查g1.yaml存在和有效
```

### Q28: 训练过程中NaN错误

**A:** 常见原因：

```
原因 → 检查项 → 修复方法

运动数据包含NaN → 加载检查gmr_data → 验证原始数据
loss爆炸 → 学习率太高 或 数据不稳定 → 降低学习率 或 检查数据质量
超参数问题 → 网络参数 → 参考官方配置调整
```

---

## 总结

**三个关键问题的答案**：

| Q | 答案 | 文档 |
|---|------|------|
| **官方数据从哪来？** | Motion Capture → GMR → Legged Lab格式 | PKL_DATA_DOCUMENTATION.md |
| **能用自己的数据吗？** | 可以！三步转换 | CUSTOM_DATASET_GUIDE.md |
| **具体怎么做？** | 见QUICK_REFERENCE.md的命令 | QUICK_REFERENCE.md |

**接下来的步骤**：

🟢 如果你想**快速开始**：用官方数据训练
🟡 如果你想**用自己的数据**：阅读CUSTOM_DATASET_GUIDE.md
🔴 如果你**遇到问题**：查这个FAQ或对应文档

祝训练顺利！🚀
