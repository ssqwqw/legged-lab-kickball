#!/usr/bin/env python3
"""
脚本：打印walk_and_run目录中所有pkl文件的内容结构
"""

import os
import pickle
from pathlib import Path
import numpy as np


def print_pkl_contents(pkl_file_path):
    """打印单个pkl文件的内容"""
    
    if not os.path.exists(pkl_file_path):
        print(f"❌ 文件不存在: {pkl_file_path}")
        return
    
    print("\n" + "=" * 80)
    print(f"📄 文件: {os.path.basename(pkl_file_path)}")
    print("=" * 80)
    
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print(f"❌ 数据不是字典类型，是: {type(data)}")
            return
        
        print(f"✅ 键数量: {len(data)}")
        print(f"✅ 所有键: {list(data.keys())}")
        print()
        
        # 打印每个键的详细信息
        for key, value in data.items():
            print(f"\n🔹 Key: '{key}'")
            print(f"   类型: {type(value).__name__}")
            
            if isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")
                print(f"   DType: {value.dtype}")
                print(f"   Min: {value.min():.6f}, Max: {value.max():.6f}")
                # 打印前3帧的数据
                if value.ndim == 1:
                    print(f"   数值: {value}")
                else:
                    print(f"   前3帧数据:")
                    for i in range(min(3, value.shape[0])):
                        if value.ndim == 2:
                            print(f"      Frame {i}: {value[i]}")
                        elif value.ndim == 3:
                            print(f"      Frame {i}: {value[i]}")
            elif isinstance(value, (int, float)):
                print(f"   值: {value}")
            elif isinstance(value, list):
                print(f"   长度: {len(value)}")
                if len(value) > 0:
                    print(f"   首个元素类型: {type(value[0]).__name__}")
                    print(f"   前3个元素: {value[:3]}")
            else:
                print(f"   值: {value}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    # walk_and_run 数据目录
    data_dir = Path("/home/user/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run")
    
    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 获取所有pkl文件
    pkl_files = sorted([f for f in data_dir.glob("*.pkl")])
    
    print(f"\n🎯 找到 {len(pkl_files)} 个pkl文件:\n")
    for f in pkl_files[:5]:  # 先打印前5个文件的名称
        print(f"   - {f.name}")
    if len(pkl_files) > 5:
        print(f"   ... 还有 {len(pkl_files) - 5} 个文件")
    
    # 打印第一个文件的详细内容
    if pkl_files:
        print_pkl_contents(str(pkl_files[0]))
        
        # 打印概览
        print("\n" + "=" * 80)
        print("📊 所有pkl文件的概览")
        print("=" * 80)
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and 'dof_pos' in data:
                    num_frames = data['dof_pos'].shape[0]
                    num_dofs = data['dof_pos'].shape[1] if len(data['dof_pos'].shape) > 1 else 1
                    fps = data.get('fps', 'N/A')
                    loop_mode = data.get('loop_mode', 'N/A')
                    
                    print(f"{pkl_file.name:50s} | Frames: {num_frames:4d} | DoFs: {num_dofs:2d} | FPS: {fps} | LoopMode: {loop_mode}")
            except Exception as e:
                print(f"{pkl_file.name:50s} | ❌ Error: {e}")
    else:
        print("❌ 没有找到pkl文件")


if __name__ == "__main__":
    main()
