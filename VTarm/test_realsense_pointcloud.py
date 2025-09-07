#!/usr/bin/env python3
"""
RealSense点云可视化测试
最简单的实现：显示RealSense点云和坐标轴
"""

from realsense.realsense import RealSense
from visualization.pointcloud_viewer import PointCloudViewer

def main():
    """主函数"""
    print("=== RealSense点云可视化测试 ===")
    # 初始化相机
    try:
        camera = RealSense(width=640, height=480, fps=30, use_color=True, non_blocking=True)
    except Exception as e:
        print(f"相机初始化失败: {e}")
        return
    
    # 初始化可视化器
    viewer = PointCloudViewer("RealSense点云显示")
    if not viewer.setup():
        camera.release()
        return

    viewer.add_coordinate_frame("origin", size=0.05)
    # 点云将按名称动态创建/更新
    try:
        while viewer.is_running():
            # 获取点云数据（60cm以内），并转换到用户坐标系：
            # X_user = Z_rs, Y_user = X_rs, Z_user = -Y_rs
            points, colors = camera.get_pointcloud(min_distance=0.05, max_distance=0.3, to_user_frame=True)
            
            if points is not None and len(points) > 0:
                # 更新/创建名为 "realsense" 的点云
                viewer.update_pointcloud("realsense", points, colors)
            
            # 更新显示（使用通用更新循环，内部处理事件并渲染）
            if not viewer.run_update_cycle(0.01):
                break
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n运行错误: {e}")
    finally:
        print("正在清理...")
        viewer.close()
        camera.release()
        print("完成")


if __name__ == "__main__":
    main()