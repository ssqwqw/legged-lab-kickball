# G1 walk_and_run pkl数据集说明

## 📊 数据集概览

`source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run` 目录包含 **30个pkl格式的动作数据文件**，用于AMP（Adversarial Motion Priors）强化学习训练。

## 📋 pkl文件的生成来源

这些pkl文件**来自GMR格式数据的转换**，转换过程通过以下脚本完成：

### 转换流程：
1. **输入格式**：GMR格式的pickle文件
   - 包含：fps, root_pos, root_rot, dof_pos 等

2. **转换脚本**：`scripts/tools/retarget/dataset_retarget.py`
   ```bash
   bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \
       --robot g1 \
       --input_dir data/gmr/ \
       --output_dir data/lab/ \
       --config_file scripts/tools/retarget/config/g1_29dof.yaml \
       --loop clamp
   ```

3. **转换内容**：
   - 从GMR DOF名称映射到Legged Lab DOF名称
   - 计算关键body的世界坐标位置
   - 设置循环模式（clamp或wrap）

## 🗂️ pkl文件包含的数据结构

每个pkl文件是一个**Python字典**，包含以下6个关键字段：

### 1️⃣ **fps** (float)
- **含义**：帧率（frames per second）
- **值**：约30 FPS（例如：29.92, 30.0）
- **用途**：计算时间步长 `dt = 1 / fps`

### 2️⃣ **root_pos** (ndarray, float64)
- **形状**：`(num_frames, 3)` - 所有帧的根部位置
- **含义**：机器人根部在世界坐标系中的XYZ位置
- **数值范围**：通常 X, Y [-2, 5], Z [0.7, 1.0] 米

```
例：Frame 0: [0.0, 0.0, 0.7647232]   # 原点，高度约0.76m
    Frame 1: [-0.00115, -0.00063, 0.7656]  # 略微移动
```

### 3️⃣ **root_rot** (ndarray, float32)
- **形状**：`(num_frames, 4)` - 所有帧的根部旋转
- **格式**：四元数 `(w, x, y, z)` - **注意：w是实部，在最前面！**
- **含义**：根部从世界坐标系到身体坐标系的旋转

```
例：Frame 0: [0.6811, 0.0121, -0.0042, 0.7320]
```

### 4️⃣ **dof_pos** (ndarray, float64)
- **形状**：`(num_frames, 29)` - 所有帧的29个关节角度
- **含义**：G1机器人的29个DOF（自由度）的关节位置（弧度）
- **数值范围**：通常 [-0.7, 1.4] 弧度
- **顺序**：对应 g1_29dof 配置中的DOF顺序

```
例：Frame 0: [0.1167, 0.0768, -0.0625, ..., 0.0940]  # 29个关节角
```

### 5️⃣ **loop_mode** (int)
- **取值**：
  - `0` = CLAMP（边界时使用最后一帧）
  - `1` = WRAP（循环播放）
- **当前数据**：全部为 `0` (CLAMP模式)

### 6️⃣ **key_body_pos** (ndarray, float32)
- **形状**：`(num_frames, 6, 3)` - 关键body的位置
- **含义**：6个关键身体部分在世界坐标系中的XYZ位置
- **关键bodies**（通常是）：
  - `[0]`: 左脚
  - `[1]`: 右脚  
  - `[2]`: 左膝
  - `[3]`: 右膝
  - `[4]`: 左肩
  - `[5]`: 右肩

```
例：Frame 0: 
  [[-0.0764, -0.0659,  0.0138]  # Body 0 (left foot)
   [ 0.0901, -0.0439,  0.0104]  # Body 1 (right foot)
   [-0.1999,  0.0489,  0.7001]  # Body 2 (left knee)
   [ 0.1898,  0.0138,  0.6876]  # Body 3 (right knee)
   [-0.1341,  0.0077,  1.0566]  # Body 4 (left shoulder)
   [ 0.1469,  0.0092,  1.0499]] # Body 5 (right shoulder)
```

## 📈 数据统计概览

| 动作名称 | 帧数 | DOF数 | FPS | 持续时间 | 循环模式 |
|---------|------|------|-----|---------|---------|
| B10_-__Walk_turn_left_45_stageii.pkl | 179 | 29 | 29.92 | 6.0s | CLAMP |
| B15_-__Walk_turn_around_stageii.pkl | 225 | 29 | 29.97 | 7.5s | CLAMP |
| C3_-_run_stageii.pkl | 81 | 29 | 29.82 | 2.7s | CLAMP |
| Walk_B4_-_Stand_to_Walk_Back_stageii.pkl | 388 | 29 | 29.94 | 13.0s | CLAMP |
| ... | ... | ... | ... | ... | ... |

**总计**：30个动作，帧数范围 66-388，总时长约 128秒

## 🔄 运行时加载方式

在训练环境中，这些pkl文件被 `MotionDataManager` 加载和使用：

```python
# 在 legged_lab/managers/motion_data_manager.py 中

# 1. 加载pkl文件
motion_data = joblib.load(motion_path)  # 读取字典

# 2. 提取数据
fps = motion_data["fps"]
root_pos = torch.from_numpy(motion_data["root_pos"]).float()
root_rot = torch.from_numpy(motion_data["root_rot"]).float()
dof_pos = torch.from_numpy(motion_data["dof_pos"]).float()
key_body_pos = torch.from_numpy(motion_data["key_body_pos"]).float()

# 3. 计算派生数据
dt = 1.0 / fps
root_vel = vel_forward_diff(root_pos, dt)          # 速度
root_ang_vel = ang_vel_from_quat_diff(root_rot, dt) # 角速度
dof_vel = vel_forward_diff(dof_pos, dt)             # 关节速度
```

## 🛠️ 配置文件

转换过程使用的配置：`scripts/tools/retarget/config/g1_29dof.yaml`

```yaml
gmr_dof_names: [...]      # GMR格式中的DOF名称列表
lab_dof_names: [...]      # Legged Lab中的DOF名称列表  
lab_key_body_names: [...]  # 关键body的名称（6个）
```

## 🎯 使用场景

这些pkl动作数据用于：

1. **AMP训练** (`g1_amp_env_cfg.py`)
   - 作为motion priors（动作先验）
   - 帮助强化学习学习自然的运动

2. **DeepMimic训练** (`g1_deepmimic_env_cfg.py`)
   - 模仿这些动作

3. **动作播放** (`scripts/play_anim.py`)
   - 直接播放or混合多个动作

## 📝 数据管理配置

在任务配置文件中引用这些数据：

```python
# 例：g1_amp_env_cfg.py

class MotionDataCfg:
    motion_data_dir = os.path.join(
        LEGGED_LAB_ROOT_DIR, 
        "data", "MotionData", 
        "g1_29dof", "amp", "walk_and_run"
    )
    
    # 定义动作权重（采样概率）
    motion_data_weights = {
        "B10_-__Walk_turn_left_45_stageii": 1.0,
        "B15_-__Walk_turn_around_stageii": 1.0,
        "C3_-_run_stageii": 1.0,
        # ... 其他动作
    }
```

## 📌 总结

- **来源**：GMR运动数据经过 `dataset_retarget.py` 转换
- **格式**：Python pickle 字典
- **关键信息**：fps, root_pos, root_rot, dof_pos, loop_mode, key_body_pos
- **用途**：为AMP/DeepMimic强化学习提供真实动作示例
- **当前数据**：30个行走和奔跑的运动序列
