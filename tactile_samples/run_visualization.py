#!/usr/bin/env python3
"""
简单的可视化启动脚本
直接运行即可可视化 bittermelon_0 数据
"""

import os
import sys
from visualize_sample_data import TactileSampleVisualizer

def main():
    # 自动检测 bittermelon_0 目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bittermelon_dir = os.path.join(current_dir, 'bittermelon_1')
    
    if not os.path.exists(bittermelon_dir):
        print(f"错误: 未找到 bittermelon_0 目录: {bittermelon_dir}")
        print("请确保脚本在 tactile_samples 目录中运行")
        return 1
    
    print("=== bittermelon_0 采样数据可视化 ===")
    print(f"数据目录: {bittermelon_dir}")
    
    try:
        # Create visualizer and run
        visualizer = TactileSampleVisualizer(bittermelon_dir)
        visualizer.run_full_visualization()
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        return 0
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
