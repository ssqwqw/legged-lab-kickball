#!/usr/bin/env python3
"""
示例：如何将你的运动数据转换为GMR格式

这个脚本展示从不同数据格式转换到GMR格式的方法
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def create_gmr_data_from_arrays(
    root_pos: np.ndarray,
    root_euler: np.ndarray,
    dof_pos: np.ndarray,
    fps: float = 30.0,
) -> dict:
    """
    从基本数组创建GMR格式数据
    
    Args:
        root_pos: 根部位置，shape (N, 3)，单位：米
        root_euler: 根部欧拉角，shape (N, 3)，单位：弧度 (roll, pitch, yaw)
        dof_pos: 关节角度，shape (N, num_dofs)，单位：弧度
        fps: 帧率
    
    Returns:
        gmr_data: GMR格式的字典
    """
    
    # 转换欧拉角为四元数 (x, y, z, w)
    rotations = R.from_euler('xyz', root_euler)
    root_rot = rotations.as_quat().astype(np.float32)  # (x, y, z, w)
    
    gmr_data = {
        'fps': fps,
        'root_pos': root_pos.astype(np.float64),
        'root_rot': root_rot,
        'dof_pos': dof_pos.astype(np.float64),
        'local_body_pos': None,
        'link_body_list': None,
    }
    
    return gmr_data


def example_create_synthetic_walk_motion() -> dict:
    """
    示例：创建一个合成的行走动作
    
    这是一个简单的循环动作示例
    """
    print("🔄 生成合成行走动作示例...")
    
    num_frames = 100
    fps = 30.0
    
    # 1. 生成根部位置（直线运动）
    x = np.linspace(0, 1.5, num_frames)  # X方向：0 -> 1.5米
    y = np.zeros(num_frames)             # Y方向：保持为0
    z = 0.75 * np.ones(num_frames)       # Z方向：高度约0.75米
    root_pos = np.stack([x, y, z], axis=1)
    
    # 2. 生成根部欧拉角（基本上保持竖直）
    roll = np.zeros(num_frames)
    pitch = np.zeros(num_frames)
    yaw = np.zeros(num_frames)
    root_euler = np.stack([roll, pitch, yaw], axis=1)
    
    # 3. 生成DOF位置（简单的周期性运动）
    num_dofs = 29
    dof_pos = np.zeros((num_frames, num_dofs))
    
    # 模拟腿部摆动（使用正弦波）
    # 关节顺序参考: left_hip_pitch, right_hip_pitch, ... (29个)
    time = np.linspace(0, 2*np.pi, num_frames)
    
    # 左腿（hip_pitch）
    dof_pos[:, 0] = 0.2 * np.sin(time)          # left_hip_pitch
    # 右腿（hip_pitch）
    dof_pos[:, 1] = 0.2 * np.sin(time + np.pi)  # right_hip_pitch，相位相反
    # 膝盖弯曲
    dof_pos[:, 9] = 0.3 + 0.1 * np.sin(2*time)   # left_knee
    dof_pos[:, 10] = 0.3 + 0.1 * np.sin(2*time + np.pi)  # right_knee
    
    # 其他关节保持静止或小幅摆动
    dof_pos[:, 2:9] = 0  # 其他腿部关节
    dof_pos[:, 11:29] = 0  # 手臂和腰部
    
    # 组合为GMR格式
    gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos, fps)
    
    print(f"✅ 生成完成")
    print(f"   帧数: {num_frames}")
    print(f"   DOF数: {num_dofs}")
    print(f"   根部位置范围: X [{root_pos[:, 0].min():.2f}, {root_pos[:, 0].max():.2f}], " +
          f"Z [{root_pos[:, 2].min():.2f}, {root_pos[:, 2].max():.2f}]")
    
    return gmr_data


def example_load_from_csv(csv_file: str) -> dict:
    """
    示例：从CSV文件加载运动数据
    
    CSV格式假设：
    root_x, root_y, root_z, root_roll, root_pitch, root_yaw, dof_0, dof_1, ..., dof_28
    """
    print(f"📖 从CSV加载数据: {csv_file}")
    
    # 使用numpy加载CSV
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)  # 跳过头行
    
    # 提取列
    root_pos = data[:, :3]             # 前3列：位置
    root_euler = data[:, 3:6]          # 接下来3列：欧拉角
    dof_pos = data[:, 6:35]            # 剩余29列：DOF
    
    fps = 30.0  # 根据你的数据修改
    
    gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos, fps)
    
    print(f"✅ CSV加载完成")
    print(f"   帧数: {len(data)}")
    
    return gmr_data


def example_load_from_dict_arrays(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    fps: float = 30.0,
) -> dict:
    """
    示例：如果你已经有了四元数形式的旋转
    
    Args:
        root_rot: 四元数 (x, y, z, w) 或 (w, x, y, z) 格式
    """
    # 如果输入是 (w, x, y, z)，先转换为 (x, y, z, w)
    # root_rot = root_rot[:, [1,2,3,0]]  # 如果需要
    
    gmr_data = {
        'fps': fps,
        'root_pos': root_pos.astype(np.float64),
        'root_rot': root_rot.astype(np.float32),
        'dof_pos': dof_pos.astype(np.float64),
        'local_body_pos': None,
        'link_body_list': None,
    }
    
    return gmr_data


def save_gmr_data(gmr_data: dict, output_file: str):
    """保存GMR数据为pickle文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(gmr_data, f)
    
    print(f"✅ 数据已保存到: {output_path}")


def validate_gmr_data(gmr_data: dict):
    """验证GMR数据的正确性"""
    print("\n🔍 验证GMR数据格式...")
    
    # 检查必要字段
    required_keys = ['fps', 'root_pos', 'root_rot', 'dof_pos']
    for key in required_keys:
        assert key in gmr_data, f"缺少必要字段: {key}"
    
    root_pos = gmr_data['root_pos']
    root_rot = gmr_data['root_rot']
    dof_pos = gmr_data['dof_pos']
    
    # 检查形状
    num_frames = root_pos.shape[0]
    assert root_pos.shape == (num_frames, 3), f"root_pos形状错误: {root_pos.shape}"
    assert root_rot.shape == (num_frames, 4), f"root_rot形状错误: {root_rot.shape}"
    assert dof_pos.shape[0] == num_frames, f"dof_pos帧数不匹配: {dof_pos.shape[0]} vs {num_frames}"
    
    # 检查数据类型
    assert root_pos.dtype == np.float64, f"root_pos应该是float64，实际: {root_pos.dtype}"
    assert root_rot.dtype == np.float32, f"root_rot应该是float32，实际: {root_rot.dtype}"
    assert dof_pos.dtype == np.float64, f"dof_pos应该是float64，实际: {dof_pos.dtype}"
    
    # 检查四元数规范化
    norms = np.linalg.norm(root_rot, axis=1)
    unnormalized_count = np.sum(np.abs(norms - 1.0) > 0.01)
    if unnormalized_count > 0:
        print(f"⚠️  警告: {unnormalized_count}个四元数未规范化，自动修正...")
        root_rot = root_rot / norms[:, np.newaxis]
        gmr_data['root_rot'] = root_rot
    
    # 检查值范围（仅警告，不强制）
    print(f"   根部位置范围:")
    print(f"     X: [{root_pos[:, 0].min():.3f}, {root_pos[:, 0].max():.3f}] 米")
    print(f"     Y: [{root_pos[:, 1].min():.3f}, {root_pos[:, 1].max():.3f}] 米")
    print(f"     Z: [{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}] 米")
    
    if root_pos[:, 2].min() < 0.5 or root_pos[:, 2].max() > 1.3:
        print(f"   ⚠️  根部高度可能不合理（正常范围: 0.6-1.0 米）")
    
    print(f"   关节角度范围:")
    print(f"     Min: {dof_pos.min():.3f} 弧度 ({np.degrees(dof_pos.min()):.1f}°)")
    print(f"     Max: {dof_pos.max():.3f} 弧度 ({np.degrees(dof_pos.max()):.1f}°)")
    
    print(f"✅ 数据验证完成")
    
    return gmr_data


def main():
    """主程序：展示三个转换示例"""
    
    print("="*80)
    print("GMR数据转换示例")
    print("="*80)
    
    # 示例1：合成数据
    print("\n" + "="*80)
    print("示例1：生成合成行走动作")
    print("="*80)
    gmr_data = example_create_synthetic_walk_motion()
    gmr_data = validate_gmr_data(gmr_data)
    save_gmr_data(gmr_data, "/tmp/synthetic_walk.pkl")
    
    # 示例2：从CSV加载（注释掉，因为没有实际文件）
    print("\n" + "="*80)
    print("示例2：从CSV文件加载")
    print("="*80)
    print("💡 使用方法:")
    print("   1. 准备CSV文件，格式: root_x, root_y, root_z, roll, pitch, yaw, dof_0, ..., dof_28")
    print("   2. 调用: gmr_data = example_load_from_csv('your_file.csv')")
    print("   3. 验证: gmr_data = validate_gmr_data(gmr_data)")
    print("   4. 保存: save_gmr_data(gmr_data, 'output.pkl')")
    
    # 示例3：从numpy数组加载（最通透的方式）
    print("\n" + "="*80)
    print("示例3：从numpy数组加载")
    print("="*80)
    print("💡 使用方法:")
    print("   # 假设你有已加载的数据")
    print("   root_pos = np.array(...)  # shape: (N, 3)")
    print("   root_euler = np.array(...)  # shape: (N, 3)")
    print("   dof_pos = np.array(...)  # shape: (N, 29)")
    print("   ")
    print("   gmr_data = create_gmr_data_from_arrays(root_pos, root_euler, dof_pos, fps=30.0)")
    print("   gmr_data = validate_gmr_data(gmr_data)")
    print("   save_gmr_data(gmr_data, 'output.pkl')")
    
    # 创建一个实际的示例用于测试
    print("\n" + "="*80)
    print("创建测试用例...")
    print("="*80)
    
    # 如果你有真实CSV文件，可以这样用：
    data_dir = Path("/home/user/legged_lab/data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    print(f"✅ 数据目录: {data_dir}")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
✅ 转换流程：
   1. 加载你的原始数据 (CSV/JSON/H5等)
   2. 提取: 根部位置(3D) + 欧拉角(3D) + 关节角度(N)
   3. 使用 create_gmr_data_from_arrays() 转换为GMR格式
   4. 使用 validate_gmr_data() 检查数据质量
   5. 使用 save_gmr_data() 保存为pickle
   6. 运行 dataset_retarget.py 转换为Legged Lab格式
   7. 在训练中使用！

💾 保存位置：/home/user/legged_lab/data/

📝 更改这个脚本以匹配你的数据格式
    """)


if __name__ == "__main__":
    main()
