#!/usr/bin/env python3
"""
重采样脚本：将运动数据从任意fps转换到30fps

使用方法：
    bash .codex/run-in-env.sh python scripts/tools/retarget/resample_fps.py \
        --input_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/fpstrans/ \
        --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/fpstrans_resampled/ \
        --target_fps 30.0
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.interpolate import interp1d


def resample_motion_data(data: dict, target_fps: float = 30.0) -> dict:
    """
    将运动数据从原始fps重采样到目标fps
    
    Args:
        data: 包含 'fps', 'root_pos', 'root_rot', 'dof_pos' 的字典
        target_fps: 目标帧率 (Hz)
    
    Returns:
        重采样后的数据字典
    """
    
    source_fps = data.get('fps')
    if source_fps is None:
        raise ValueError("Motion data must contain 'fps' field")
    
    if abs(source_fps - target_fps) < 0.01:
        print(f"  已是 {target_fps} fps，无需重采样")
        return data
    
    num_frames = len(data['root_pos'])
    
    # 源时间戳（秒）
    source_times = np.arange(num_frames) / source_fps
    
    # 目标时间戳：保持总时长一致
    total_duration = source_times[-1]
    target_num_frames = int(np.round(total_duration * target_fps)) + 1
    target_times = np.arange(target_num_frames) / target_fps
    
    # 确保目标时间戳不超出源范围
    target_times = target_times[target_times <= source_times[-1]]
    
    print(f"  源: {num_frames} 帧 @ {source_fps:.2f} fps (总时长 {total_duration:.3f}s)")
    print(f"  目标: {len(target_times)} 帧 @ {target_fps:.2f} fps")
    
    # 构建插值函数（线性）
    # root_pos: (N, 3)
    root_pos_interp = interp1d(
        source_times, data['root_pos'], axis=0, kind='linear',
        fill_value='extrapolate', bounds_error=False
    )
    new_root_pos = root_pos_interp(target_times)
    
    # root_rot: (N, 4) - 四元数需要特殊处理（SLERP）
    # 为了简单起见，这里也用线性插值，但实际上应该用 SLERP
    # TODO: 如果质量要求高，可以用 scipy.spatial.transform.Rotation.slerp
    root_rot_interp = interp1d(
        source_times, data['root_rot'], axis=0, kind='linear',
        fill_value='extrapolate', bounds_error=False
    )
    new_root_rot = root_rot_interp(target_times)
    
    # 重新规范化四元数（因为线性插值会破坏单位四元数约束）
    root_rot_norm = np.linalg.norm(new_root_rot, axis=1, keepdims=True)
    new_root_rot = new_root_rot / (root_rot_norm + 1e-8)
    
    # dof_pos: (N, 29)
    dof_pos_interp = interp1d(
        source_times, data['dof_pos'], axis=0, kind='linear',
        fill_value='extrapolate', bounds_error=False
    )
    new_dof_pos = dof_pos_interp(target_times)
    
    # 构建新数据
    resampled_data = {
        'fps': float(target_fps),
        'root_pos': new_root_pos.astype(np.float64),
        'root_rot': new_root_rot.astype(np.float32),
        'dof_pos': new_dof_pos.astype(np.float64),
        'local_body_pos': data.get('local_body_pos'),
        'link_body_list': data.get('link_body_list'),
    }
    
    # 保留其他字段（如果有）
    for key in data:
        if key not in resampled_data:
            resampled_data[key] = data[key]
    
    return resampled_data


def main():
    parser = argparse.ArgumentParser(
        description="将运动数据从任意fps重采样到目标fps"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="输入目录（包含 .pkl 文件）"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="输出目录（保存重采样后的 .pkl 文件）"
    )
    parser.add_argument(
        '--target_fps',
        type=float,
        default=30.0,
        help="目标帧率（Hz），默认 30.0"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_fps = args.target_fps
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 pkl 文件
    pkl_files = sorted(input_dir.glob('*.pkl'))
    
    if not pkl_files:
        print(f"❌ 未在 {input_dir} 中找到 .pkl 文件")
        return
    
    print(f"📂 找到 {len(pkl_files)} 个 pkl 文件")
    print(f"🎯 目标帧率: {target_fps} fps")
    print()
    
    success_count = 0
    skip_count = 0
    
    for pkl_file in pkl_files:
        print(f"处理: {pkl_file.name}")
        try:
            # 加载数据
            data = joblib.load(pkl_file)
            
            # 重采样
            resampled_data = resample_motion_data(data, target_fps)
            
            # 保存
            output_file = output_dir / pkl_file.name
            joblib.dump(resampled_data, output_file)
            
            print(f"  ✅ 保存到: {output_file}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        
        print()
    
    # 统计
    print("=" * 60)
    print(f"✅ 成功: {success_count}/{len(pkl_files)}")
    if skip_count > 0:
        print(f"⏭️  跳过: {skip_count}/{len(pkl_files)}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
    print("\n下一步:")
    print(f"1. 检查输出文件是否正确")
    print(f"2. 运行 dataset_retarget.py 进行进一步处理:")
    print(f"   bash .codex/run-in-env.sh python scripts/tools/retarget/dataset_retarget.py \\")
    print(f"       --robot g1 \\")
    print(f"       --input_dir {output_dir} \\")
    print(f"       --output_dir source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/my_data/ \\")
    print(f"       --config_file scripts/tools/retarget/config/g1_29dof.yaml \\")
    print(f"       --loop clamp")


if __name__ == '__main__':
    main()
