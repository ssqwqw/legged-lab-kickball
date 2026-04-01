# 使用自己的运动数据集指南

## 🎯 概述

**可以的！** 你可以使用自己的运动数据。流程是：

```
你的原始动作数据 (CSV/BVH/其他格式)
         ↓
   转换为 GMR 格式 (pickle)
         ↓
   运行 dataset_retarget.py
         ↓
   输出 Legged Lab 格式 (pickle)
         ↓
   在训练中使用
```

## 📦 第1步：准备数据为 GMR 格式

### GMR 格式定义

你的数据需要转换为一个 **pickle 字典**，包含以下字段：

```python
gmr_data = {
    'fps': 30.0,                              # 帧率 (float)
    'root_pos': np.ndarray(shape=(N, 3)),    # 根部XYZ位置 (float64)
    'root_rot': np.ndarray(shape=(N, 4)),    # 根部四元数 (x,y,z,w) (float32)
    'dof_pos': np.ndarray(shape=(N, 29)),    # 29个DOF角度 (float64)
    'local_body_pos': None,                  # 可选，暂时未使用
    'link_body_list': None,                  # 可选，暂时未使用
}
```

### 重要细节

| 字段 | 格式 | 说明 |
|------|------|------|
| **fps** | float | 帧率，例如 30.0, 29.97 等 |
| **root_pos** | (N, 3) float64 | 根部在世界坐标系中的位置（米）<br>X, Y: 水平位置，Z: 高度（约0.7-1.0米） |
| **root_rot** | (N, 4) float32 | 四元数格式 **(x, y, z, w)** ⚠️ 注意：w在最后！<br>从世界坐标系到身体坐标系的旋转 |
| **dof_pos** | (N, 29) float64 | 29个关节角度（弧度）<br>**顺序必须与gmr_dof_names一致** |

### 数据值范围参考

```
root_pos:
  - X轴: [-3, 5] 米
  - Y轴: [-1, 5] 米  
  - Z轴: [0.6, 1.0] 米（身体高度）

root_rot (四元数，已规范化):
  - 每个分量: [-1, 1]
  - 满足: x² + y² + z² + w² = 1

dof_pos (关节角度):
  - 典型范围: [-1.5, 1.5] 弧度
  - 对应: [-86°, 86°]
```

### 转换脚本示例

创建 `convert_your_data_to_gmr.py`：

```python
import pickle
import numpy as np
from pathlib import Path

def convert_your_data_to_gmr(
    input_file: str,
    output_file: str,
    fps: float = 30.0
):
    """
    将你的数据转换为GMR格式
    
    你需要根据你的数据格式修改这个函数
    """
    
    # ⬇️ 这里需要你自己实现，加载你的原始数据
    # 例如：从CSV/JSON/H5/BVH等格式读取
    
    root_pos = np.array(...)      # shape: (N, 3), float64
    root_rot = np.array(...)      # shape: (N, 4), float32，格式: (x,y,z,w)
    dof_pos = np.array(...)       # shape: (N, 29), float64
    
    # 构建GMR字典
    gmr_data = {
        'fps': fps,
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'local_body_pos': None,
        'link_body_list': None,
    }
    
    # 保存为pickle
    with open(output_file, 'wb') as f:
        pickle.dump(gmr_data, f)
    
    print(f"✅ 数据已保存到: {output_file}")
    print(f"   根部位置形状: {root_pos.shape}")
    print(f"   关节角度形状: {dof_pos.shape}")


# 使用示例
if __name__ == "__main__":
    # 将这里改为你的输入格式
    convert_your_data_to_gmr(
        input_file="your_motion.csv",      # 或其他格式
        output_file="your_motion.pkl",
        fps=30.0
    )
```

## 📋 第2步：创建或修改配置文件

### 配置文件格式 (.yaml)

需要创建一个配置文件来定义 DOF 的映射。参考：
`scripts/tools/retarget/config/g1_29dof.yaml`

### G1机器人的29个DOF顺序

**GMR格式(输入顺序)**：
```yaml
gmr_dof_names:
  - left_hip_pitch_joint       # 0
  - left_hip_roll_joint        # 1
  - left_hip_yaw_joint         # 2
  - left_knee_joint            # 3
  - left_ankle_pitch_joint     # 4
  - left_ankle_roll_joint      # 5
  - right_hip_pitch_joint      # 6
  - right_hip_roll_joint       # 7
  - right_hip_yaw_joint        # 8
  - right_knee_joint           # 9
  - right_ankle_pitch_joint    # 10
  - right_ankle_roll_joint     # 11
  - waist_yaw_joint            # 12
  - waist_roll_joint           # 13
  - waist_pitch_joint          # 14
  - left_shoulder_pitch_joint  # 15
  - left_shoulder_roll_joint   # 16
  - left_shoulder_yaw_joint    # 17
  - left_elbow_joint           # 18
  - left_wrist_roll_joint      # 19
  - left_wrist_pitch_joint     # 20
  - left_wrist_yaw_joint       # 21
  - right_shoulder_pitch_joint # 22
  - right_shoulder_roll_joint  # 23
  - right_shoulder_yaw_joint   # 24
  - right_elbow_joint          # 25
  - right_wrist_roll_joint     # 26
  - right_wrist_pitch_joint    # 27
  - right_wrist_yaw_joint      # 28
```

**Legged Lab格式(输出/训练顺序)**：
```yaml
lab_dof_names:
  - left_hip_pitch_joint       # 0
  - right_hip_pitch_joint      # 1
  - waist_yaw_joint            # 2
  - left_hip_roll_joint        # 3
  - right_hip_roll_joint       # 4
  - waist_roll_joint           # 5
  - left_hip_yaw_joint         # 6
  - right_hip_yaw_joint        # 7
  - waist_pitch_joint          # 8
  - left_knee_joint            # 9
  - right_knee_joint           # 10
  - left_shoulder_pitch_joint  # 11
  - right_shoulder_pitch_joint # 12
  - left_ankle_pitch_joint     # 13
  - right_ankle_pitch_joint    # 14
  - left_shoulder_roll_joint   # 15
  - right_shoulder_roll_joint  # 16
  - left_ankle_roll_joint      # 17
  - right_ankle_roll_joint     # 18
  - left_shoulder_yaw_joint    # 19
  - right_shoulder_yaw_joint   # 20
  - left_elbow_joint           # 21
  - right_elbow_joint          # 22
  - left_wrist_roll_joint      # 23
  - right_wrist_roll_joint     # 24
  - left_wrist_pitch_joint     # 25
  - right_wrist_pitch_joint    # 26
  - left_wrist_yaw_joint       # 27
  - right_wrist_yaw_joint      # 28
```

**关键body定义**：
```yaml
lab_key_body_names:
  - left_ankle_roll_link      # 0 - 左脚
  - right_ankle_roll_link     # 1 - 右脚
  - left_wrist_yaw_link       # 2 - 左手
  - right_wrist_yaw_link      # 3 - 右手
  - left_shoulder_roll_link   # 4 - 左肩
  - right_shoulder_roll_link  # 5 - 右肩
```

### 如果使用其他机器人

修改 `gmr_dof_names` 和 `lab_dof_names` 的顺序即可。

## 🚀 第3步：运行转换脚本

### 选项A：单个文件转换

```bash
cd /home/user/legged_lab

bash .codex/run-in-env.sh python scripts/tools/retarget/single_retarget.py \
    --robot g1 \
    --input_file data/my_data/my_walk.pkl \
    --output_file data/converted/my_walk.pkl \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --loop clamp \
    --headless
```

**参数说明**：
- `--input_file`: 你的 GMR 格式 pkl 文件
- `--output_file`: 输出位置
- `--config_file`: DOF 映射配置文件
- `--loop {clamp|wrap}`: 
  - `clamp`: 到达末尾时停留在最后一帧（默认）
  - `wrap`: 循环播放
- `--frame_range START END`: 可选，只转换特定帧范围

### 选项B：批量转换多个文件

```bash
cd /home/user/legged_lab

bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 \
    --input_dir data/my_gmr_motions/ \
    --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_custom_motions/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --loop clamp
```

这会自动：
1. 扫描 `input_dir` 中的所有 `.pkl` 文件
2. 一次性在模拟器中加载所有运动（高效！）
3. 计算关键body位置
4. 保存转换后的数据

## 📂 第4步：在训练中使用自己的数据

### 更新配置文件

编辑 `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`：

```python
class MotionDataCfg:
    # 改为你的数据目录
    motion_data_dir = os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "data", "MotionData", "g1_29dof", "amp", "my_custom_motions"  # ← 改这里
    )
    
    # 定义动作名称和权重（采样概率）
    motion_data_weights = {
        "my_walk_forward": 2.0,      # 行走，权重2.0
        "my_walk_backward": 1.0,     # 后退，权重1.0
        "my_run_forward": 2.0,       # 跑步，权重2.0
        # ... 其他动作
    }
```

### 开始训练

```bash
cd /home/user/legged_lab

bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless
```

## ⚠️ 常见问题

### Q1: 我的数据是 BVH/C3D 格式怎么办？

A: 需要先转换为可以读取的格式（如 numpy 数组），然后：
1. 实现一个转换脚本读取你的格式
2. 提取：根部位置、四元数、关节角度
3. 保存为 GMR pickle 格式

例如 BVH:
```python
import bvh  # 需要安装 bvh 库
frames = bvh_data.get_frames()  # 获取帧数据
root_pos = np.array([frame[0:3] for frame in frames])  # 提取位置
# ... 等等
```

### Q2: 我的数据中没有根部四元数怎么办？

A: 如果你只有根部欧拉角（roll, pitch, yaw），使用这段代码转换：

```python
from scipy.spatial.transform import Rotation as R
import numpy as np

# 假设你有欧拉角 (roll, pitch, yaw)
euler_angles = np.array([...])  # shape: (N, 3), 单位：弧度

# 转换为四元数 (x, y, z, w)
rotations = R.from_euler('xyz', euler_angles)
root_rot = rotations.as_quat()  # 返回 (x, y, z, w) 格式，正好！
```

### Q3: 我的关节角度顺序不一样怎么办？

A: 不用担心，这就是 `gmr_dof_names` 配置的作用。在配置文件中更改顺序即可。

例如如果你的数据是 `[右脚踝, 左脚踝, ...]`，就在配置文件中这样定义：
```yaml
gmr_dof_names:
  - right_ankle_pitch_joint  # 你的第0个
  - left_ankle_pitch_joint   # 你的第1个
  # ... 等等
```

### Q4: 数据质量差会影响训练吗？

A: 会的。好的运动数据应该：
- ✅ 运动自然流畅（无抖动跳跃）
- ✅ 根部高度保持在 [0.6, 1.0] 米
- ✅ 关节角度在机器人物理限制范围内
- ✅ 帧率一致（通常 30 FPS）

### Q5: 需要多少条运动数据？

A: 取决于使用场景：
- **最少**：3-5条基本动作（走、跑、转身）
- **推荐**：15-30条多样化动作（官方数据集规模）
- **理想**：50+条 + 多个机器人的数据

## 📊 数据验证脚本

运行这个脚本检查你转换的数据质量：

```python
import pickle
import numpy as np

def validate_gmr_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    assert 'fps' in data, "缺少 fps"
    assert 'root_pos' in data, "缺少 root_pos"
    assert 'root_rot' in data, "缺少 root_rot"
    assert 'dof_pos' in data, "缺少 dof_pos"
    
    root_pos = data['root_pos']
    root_rot = data['root_rot']
    dof_pos = data['dof_pos']
    
    print(f"✅ 基本结构检查通过")
    print(f"   根部位置: {root_pos.shape}")
    print(f"   根部旋转: {root_rot.shape}")
    print(f"   关节角度: {dof_pos.shape}")
    
    # 检查四元数规范化
    norms = np.linalg.norm(root_rot, axis=1)
    assert np.allclose(norms, 1.0), "四元数未规范化！"
    
    # 检查值范围
    assert root_pos[:, 2].min() > 0.5, "根部高度太低"
    assert root_pos[:, 2].max() < 1.3, "根部高度太高"
    
    print(f"✅ 所有检查通过！")

# 使用
validate_gmr_data("data/my_data/motion.pkl")
```

## 🎓 完整工作流示例

```bash
# 1. 准备数据（转换为GMR格式）
python convert_your_data_to_gmr.py

# 2. 创建配置文件（如果需要）或使用默认配置

# 3. 验证数据
python check_gmr_format.py

# 4. 转换为Legged Lab格式
bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 \
    --input_dir data/my_gmr_motions/ \
    --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_data/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml \
    --loop clamp

# 5. 更新训练配置中的motion_data_dir

# 6. 开始训练！
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
    --task g1_amp_env_cfg \
    --headless
```

## 📖 相关文件位置

- **转换脚本**：`scripts/tools/retarget/`
  - `gmr_to_lab.py` - 核心转换函数
  - `single_retarget.py` - 单文件转换
  - `dataset_retarget.py` - 批量转换

- **配置文件**：`scripts/tools/retarget/config/`
  - `g1_29dof.yaml` - 当前使用的G1配置

- **训练配置**：`source/legged_lab/legged_lab/tasks/locomotion/`
  - `amp/config/g1/g1_amp_env_cfg.py` - AMP训练配置

- **数据位置**：`source/legged_lab/legged_lab/data/MotionData/`
  - 放置你转换的数据

祝你成功整合自己的运动数据！🚀
