#!/usr/bin/env python3
"""
示例：如何加载和使用pkl运动数据

这个脚本演示：
1. 如何加载pkl文件
2. 如何提取和理解运动数据
3. 如何计算衍生数据（速度、角速度等）
"""

import os
import pickle
import numpy as np
from pathlib import Path


def load_motion_data(pkl_file_path):
    """加载单个pkl运动数据文件"""
    with open(pkl_file_path, 'rb') as f:
        return pickle.load(f)


def compute_velocities(positions, fps):
    """
    使用前向差分计算速度
    
    Args:
        positions: 位置数组，shape (num_frames, n_dims)
        fps: 帧率
    
    Returns:
        velocities: 速度数组，shape (num_frames, n_dims)
    """
    dt = 1.0 / fps
    # 前向差分：v[t] = (p[t+1] - p[t]) / dt
    velocities = np.zeros_like(positions)
    velocities[:-1] = (positions[1:] - positions[:-1]) / dt
    # 最后一帧速度等于倒数第二帧
    velocities[-1] = velocities[-2]
    return velocities


def quaternion_to_angle_axis(quat):
    """
    将四元数转换为角-轴表示
    
    Args:
        quat: 四元数，格式 (w, x, y, z)
    
    Returns:
        angle, axis: 旋转角度（弧度），旋转轴
    """
    w, x, y, z = quat
    # 规范化四元数
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 0, np.array([0, 0, 1])
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 转换为角-轴
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    sin_half_angle = np.sqrt(1 - w*w)
    
    if sin_half_angle < 1e-6:
        axis = np.array([0, 0, 1])
    else:
        axis = np.array([x, y, z]) / sin_half_angle
    
    return angle, axis


def analyze_motion_data(pkl_file_path):
    """分析单个运动数据文件"""
    
    motion_data = load_motion_data(pkl_file_path)
    file_name = os.path.basename(pkl_file_path)
    
    print("\n" + "="*100)
    print(f"📊 运动数据分析: {file_name}")
    print("="*100)
    
    # 基本信息
    fps = motion_data["fps"]
    loop_mode = motion_data["loop_mode"]
    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"]
    dof_pos = motion_data["dof_pos"]
    key_body_pos = motion_data["key_body_pos"]
    
    num_frames = len(root_pos)
    duration = (num_frames - 1) / fps
    
    print(f"\n⏱️  时间信息:")
    print(f"   帧率 (FPS):      {fps:.2f}")
    print(f"   总帧数:         {num_frames}")
    print(f"   动作持续时间:    {duration:.2f} 秒")
    print(f"   时间步长 (dt):   {1/fps:.4f} 秒")
    
    print(f"\n🤖 机器人配置:")
    print(f"   关节数 (DOF):    {dof_pos.shape[1]}")
    print(f"   关键body数:     {key_body_pos.shape[1]}")
    print(f"   循环模式:       {'WRAP' if loop_mode == 1 else 'CLAMP'}")
    
    # 根部位置分析
    print(f"\n📍 根部位置统计 (世界坐标系):")
    print(f"   X轴: {root_pos[:, 0].min():.3f} ~ {root_pos[:, 0].max():.3f} 米")
    print(f"   Y轴: {root_pos[:, 1].min():.3f} ~ {root_pos[:, 1].max():.3f} 米")
    print(f"   Z轴: {root_pos[:, 2].min():.3f} ~ {root_pos[:, 2].max():.3f} 米 (高度)")
    
    # 计算根部速度
    root_vel = compute_velocities(root_pos, fps)
    root_speed = np.linalg.norm(root_vel, axis=1)
    
    print(f"\n🏃 运动速度:")
    print(f"   平均速度:       {root_speed.mean():.3f} m/s")
    print(f"   最大速度:       {root_speed.max():.3f} m/s")
    print(f"   速度类型:       {'行走' if root_speed.max() < 2.0 else '跑步'}")
    
    # 关节角度分析
    print(f"\n🦴 关节角度统计 (所有{dof_pos.shape[1]}个DOF):")
    print(f"   最小角度:       {dof_pos.min():.3f} 弧度 ({np.degrees(dof_pos.min()):.1f}°)")
    print(f"   最大角度:       {dof_pos.max():.3f} 弧度 ({np.degrees(dof_pos.max()):.1f}°)")
    print(f"   角度范围:       {dof_pos.max() - dof_pos.min():.3f} 弧度")
    
    # 根部旋转分析
    print(f"\n🔄 根部旋转 (四元数 w, x, y, z):")
    quat_first = root_rot[0]
    quat_last = root_rot[-1]
    angle_first, axis_first = quaternion_to_angle_axis(quat_first)
    angle_last, axis_last = quaternion_to_angle_axis(quat_last)
    
    print(f"   初始位姿: {quat_first} (角度: {np.degrees(angle_first):.1f}°)")
    print(f"   最终位姿: {quat_last} (角度: {np.degrees(angle_last):.1f}°)")
    print(f"   旋转变化: {np.degrees(angle_last):.1f}°")
    
    # 关键body位置
    print(f"\n👣 关键body位置 (6个关键点):")
    body_names = ["左脚", "右脚", "左膝", "右膝", "左肩", "右肩"]
    for i, (name, positions) in enumerate(zip(body_names, key_body_pos.transpose(1, 0, 2))):
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        z_range = z_max - z_min
        print(f"   [{i}] {name:5s}: Z范围 {z_min:.3f}~{z_max:.3f}m (变化: {z_range:.3f}m)")
    
    # 示例帧数据
    print(f"\n📋 示例帧记录:")
    print(f"\n   首帧 (Frame 0):")
    print(f"   - 根部位置: {root_pos[0]} m")
    print(f"   - 根部四元数: {root_rot[0]} (w, x, y, z)")
    print(f"   - 关键关节 (DOF[0-5]): {dof_pos[0, :6]}")
    
    if num_frames > 1:
        print(f"\n   末帧 (Frame {num_frames-1}):")
        print(f"   - 根部位置: {root_pos[-1]} m")
        print(f"   - 根部四元数: {root_rot[-1]} (w, x, y, z)")
        print(f"   - 关键关节 (DOF[0-5]): {dof_pos[-1, :6]}")


def main():
    # 数据目录
    data_dir = Path("/home/user/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run")
    
    # 获取所有pkl文件
    pkl_files = sorted([f for f in data_dir.glob("*.pkl")])
    
    if not pkl_files:
        print("❌ 没有找到pkl文件")
        return
    
    # 分析前3个文件作为示例
    print(f"\n🎯 找到 {len(pkl_files)} 个pkl文件，分析前3个...")
    
    for pkl_file in pkl_files[:3]:
        try:
            analyze_motion_data(str(pkl_file))
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结统计
    print("\n" + "="*100)
    print("📈 全数据集统计")
    print("="*100)
    
    total_frames = 0
    total_duration = 0
    walk_count = 0
    run_count = 0
    
    for pkl_file in pkl_files:
        try:
            data = load_motion_data(pkl_file)
            num_frames = len(data["root_pos"])
            fps = data["fps"]
            duration = (num_frames - 1) / fps
            total_frames += num_frames
            total_duration += duration
            
            # 判断行走还是跑步
            root_vel = compute_velocities(data["root_pos"], fps)
            max_speed = np.linalg.norm(root_vel, axis=1).max()
            
            if max_speed < 2.0:
                walk_count += 1
            else:
                run_count += 1
        except Exception as e:
            pass
    
    print(f"\n总计:")
    print(f"   数据集中的动作数:   {len(pkl_files)}")
    print(f"   行走动作:         {walk_count}")
    print(f"   跑步动作:         {run_count}")
    print(f"   总帧数:          {total_frames:,}")
    print(f"   总时长:          {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
    print(f"   平均动作时长:     {total_duration/len(pkl_files):.1f} 秒")


if __name__ == "__main__":
    main()
