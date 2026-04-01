# 使用自己数据集的快速参考

## 📌 三句话总结

**可以！** 转换过程：**你的数据 → GMR格式 → Legged Lab格式 → 训练**

## 🔄 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│ 你的原始数据 (CSV/BVH/Motion Capture/其他)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤1：转换为GMR格式 (pickle)                                 │
│ 需要: root_pos(3), root_rot(4,四元数), dof_pos(29)           │
│ 使用: example_gmr_conversion.py                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 参考配置文件                                                  │
│ scripts/tools/retarget/config/g1_29dof.yaml                 │
│ (定义DOF映射关系)                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤2：运行转换脚本                                            │
│ bash .codex/run-in-env.sh python                            │
│   scripts/tools/retarget/dataset_retarget.py \              │
│   --input_dir data/my_gmr/ \                                │
│   --output_dir data/MotionData/g1_29dof/amp/my_data/ \      │
│   --config_file scripts/tools/retarget/config/g1_29dof.yaml │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤3：更新训练配置                                            │
│ motion_data_dir = .../my_data/                              │
│ motion_data_weights = {                                     │
│     "my_motion_1": 1.0,                                     │
│     "my_motion_2": 1.0, ...                                 │
│ }                                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤4：开始训练                                               │
│ bash .codex/run-in-env.sh python                            │
│   scripts/rsl_rl/train.py --task g1_amp_env_cfg --headless  │
└─────────────────────────────────────────────────────────────┘
```

## 💾 GMR数据格式速查

必须是 **Python pickle 字典**，包含：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **fps** | float | 标量 | 30.0 |
| **root_pos** | ndarray | (N, 3) | float64 - 根部位置(米) |
| **root_rot** | ndarray | (N, 4) | float32 - 四元数(x,y,z,w) ⭐️ |
| **dof_pos** | ndarray | (N, 29) | float64 - 关节角度(弧度) |
| optional: local_body_pos | - | - | None |
| optional: link_body_list | - | - | None |

**⭐️ 重要**：四元数格式是 **(x, y, z, w)**，不是 (w, x, y, z)！

## 📝 三种数据制备方法

### 方法1：你有CSV文件

```python
import numpy as np
from example_gmr_conversion import create_gmr_data_from_arrays, validate_gmr_data, save_gmr_data

# 加载CSV
data = np.loadtxt('my_motion.csv', delimiter=',', skiprows=1)
root_pos = data[:, :3]
root_euler = data[:, 3:6]  # roll, pitch, yaw
dof_pos = data[:, 6:35]

# 转换
gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos)
gmr_data = validate_gmr_data(gmr_data)
save_gmr_data(gmr_data, 'my_motion.pkl')
```

### 方法2：你有H5/NPZ/其他格式

```python
import h5py
from example_gmr_conversion import create_gmr_data_from_arrays, validate_gmr_data, save_gmr_data

# 从H5加载
with h5py.File('my_motion.h5', 'r') as f:
    root_pos = f['root_pos'][:]
    root_euler = f['root_euler'][:]
    dof_pos = f['dof_pos'][:]

# 转换
gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos)
save_gmr_data(gmr_data, 'my_motion.pkl')
```

### 方法3：你有BVH文件（Motion Capture）

```bash
# 安装BVH库
bash .codex/run-in-env.sh pip install bvh

# 然后在Python中：
import bvh
from example_gmr_conversion import ...

with open('motion.bvh', 'r') as f:
    mocap = bvh.parse(f)
    frames = mocap.frames
    
# 提取根部位置和关节角度
root_pos = frames[:, :3]  # 根部位置
dof_pos = frames[:, 6:]   # 关节角度（跳过根部的6DOF）
# ... 等等
```

## ⚙️ 四元数转换速查

### 如果你有欧拉角 (roll, pitch, yaw)

```python
from scipy.spatial.transform import Rotation as R

euler = np.array([roll, pitch, yaw])  # 形状: (N, 3)
quat = R.from_euler('xyz', euler).as_quat()  # ✅ 返回 (x,y,z,w)
```

### 如果你有旋转矩阵

```python
from scipy.spatial.transform import Rotation as R

rotation_matrix = np.array([...])  # 形状: (3, 3)
quat = R.from_matrix(rotation_matrix).as_quat()  # ✅ (x,y,z,w)
```

### 如果四元数格式是 (w,x,y,z)，需要重新排列

```python
quat_wxyz = np.array([...])  # 输入: (w,x,y,z)
quat_xyzw = quat_wxyz[:, [1,2,3,0]]  # 输出: (x,y,z,w) ✅
```

## 🚀 快速命令

### 1️⃣ 准备数据（使用example_gmr_conversion.py）

```bash
cd /home/user/legged_lab
bash .codex/run-in-env.sh python example_gmr_conversion.py
```

### 2️⃣ 转换GMR→Legged Lab（单个文件）

```bash
bash .codex/run-in-env.sh python scripts/tools/retarget/single_retarget.py \
    --robot g1 \
    --input_file data/my_gmr/motion.pkl \
    --output_file data/my_converted/motion.pkl \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --loop clamp
```

### 3️⃣ 转换多个文件（批量）

```bash
bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 \
    --input_dir data/my_gmr/ \
    --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_motions/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --loop clamp
```

### 4️⃣ 验证转换结果

```bash
bash .codex/run-in-env.sh python print_pkl_contents.py
bash .codex/run-in-env.sh python analyze_motion_data.py
```

## 📂 重要文件位置参考

```
/home/user/legged_lab/
├── scripts/tools/retarget/
│   ├── gmr_to_lab.py              ← 核心转换函数
│   ├── single_retarget.py          ← 单文件转换脚本
│   ├── dataset_retarget.py         ← 批量转换脚本
│   └── config/
│       └── g1_29dof.yaml           ← DOF映射配置
│
├── example_gmr_conversion.py        ← 数据转换示例 ⭐️
├── CUSTOM_DATASET_GUIDE.md          ← 详细指南 ⭐️
│
├── source/legged_lab/legged_lab/
│   ├── data/MotionData/
│   │   └── g1_29dof/amp/
│   │       ├── walk_and_run/       ← 官方数据
│   │       └── [你的数据目录]/      ← 放这里
│   │
│   └── tasks/locomotion/amp/config/g1/
│       └── g1_amp_env_cfg.py       ← 修改motion_data_dir
│
└── scripts/rsl_rl/
    └── train.py                    ← 开始训练
```

## ✅ 检查清单

- [ ] 数据转换为GMR格式 (pickle字典)
- [ ] 验证数据形状：root_pos(N,3), root_rot(N,4), dof_pos(N,29)
- [ ] 验证root_rot是(x,y,z,w)格式，已规范化
- [ ] 数据值在合理范围内（高度0.6-1.0m，角度-1.5~1.5rad）
- [ ] 配置文件(yaml)中DOF顺序正确
- [ ] 运行dataset_retarget.py转换
- [ ] 更新训练配置中的motion_data_dir
- [ ] 更新motion_data_weights字典
- [ ] 开始训练！

## 🆘 常见问题

**Q: 转换报错说DOF不匹配？**
A: 检查你的配置文件中的gmr_dof_names顺序是否与你的数据顺序一致

**Q: 四元数相关的错误？**
A: 确保四元数是(x,y,z,w)格式，且已规范化（norm=1.0）

**Q: 模拟器崩溃或动作看起来奇怪？**
A: 检查关节角度是否超出物理限制，根部高度是否在0.6-1.0m范围

**Q: 训练不收敛？**
A: 可能是数据质量问题，尝试使用官方数据对比，或增加数据多样性

## 📚 了解更多

详细文档：[CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)

示例脚本：[example_gmr_conversion.py](example_gmr_conversion.py)

官方PKL分析：[PKL_DATA_DOCUMENTATION.md](PKL_DATA_DOCUMENTATION.md)
