#!/usr/bin/env python3
"""
触觉采样数据可视化工具
用于可视化 tactile_samples 目录中的采样数据，包括：
1. 四张触觉图像的动画播放（左右rectify、左右difference）
2. 夹爪日志的曲线图（命令值vs反馈值）
3. 带接触标记的物体点云可视化

使用方法:
python visualize_sample_data.py <采样目录路径>
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
from typing import List, Tuple, Optional
import argparse
import glob

# 设置matplotlib使用英文显示，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("警告: 未安装open3d，将跳过3D点云可视化")
    HAS_OPEN3D = False


class TactileSampleVisualizer:
    """触觉采样数据可视化器"""
    
    def __init__(self, sample_dir: str):
        self.sample_dir = sample_dir
        self.sample_name = os.path.basename(sample_dir)
        
        # 数据路径
        self.left_rectify_dir = os.path.join(sample_dir, 'left_rectify')
        self.left_difference_dir = os.path.join(sample_dir, 'left_difference')
        self.right_rectify_dir = os.path.join(sample_dir, 'right_rectify')
        self.right_difference_dir = os.path.join(sample_dir, 'right_difference')
        self.gripper_log_path = os.path.join(sample_dir, 'gripper_log.csv')
        self.object_points_path = os.path.join(sample_dir, 'object_points_with_contact.npy')
        self.meta_path = os.path.join(sample_dir, 'meta.txt')
        
        # 验证数据完整性
        self._validate_data()
        
        # 加载数据
        self.gripper_data = self._load_gripper_log()
        self.object_points = self._load_object_points()
        self.image_timestamps = self._get_image_timestamps()
        
        print(f"[可视化] 加载采样数据: {self.sample_name}")
        print(f"  - 夹爪日志: {len(self.gripper_data)} 条记录")
        print(f"  - 触觉图像: {len(self.image_timestamps)} 帧")
        if self.object_points is not None:
            print(f"  - 物体点云: {self.object_points.shape[0]} 个点")
    
    def _validate_data(self):
        """验证数据文件是否存在"""
        required_dirs = [
            self.left_rectify_dir, self.left_difference_dir,
            self.right_rectify_dir, self.right_difference_dir
        ]
        required_files = [self.gripper_log_path]
        
        for dir_path in required_dirs:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"缺少目录: {dir_path}")
        
        for file_path in required_files:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"缺少文件: {file_path}")
    
    def _load_gripper_log(self) -> pd.DataFrame:
        """加载夹爪日志数据"""
        try:
            df = pd.read_csv(self.gripper_log_path)
            # 将时间戳转换为秒
            df['time_s'] = df['timestamp_ms'] / 1000.0
            return df
        except Exception as e:
            raise RuntimeError(f"加载夹爪日志失败: {e}")
    
    def _load_object_points(self) -> Optional[np.ndarray]:
        """加载物体点云数据"""
        if not os.path.exists(self.object_points_path):
            print("警告: 未找到物体点云文件")
            return None
        
        try:
            points = np.load(self.object_points_path)
            if points.shape[1] != 4:
                print(f"警告: 点云数据维度不正确，期望4维，实际{points.shape[1]}维")
                return None
            return points
        except Exception as e:
            print(f"警告: 加载物体点云失败: {e}")
            return None
    
    def _get_image_timestamps(self) -> List[int]:
        """获取所有图像的时间戳"""
        # 从left_rectify目录获取时间戳
        png_files = glob.glob(os.path.join(self.left_rectify_dir, '*.png'))
        timestamps = []
        
        for png_file in png_files:
            filename = os.path.basename(png_file)
            try:
                timestamp = int(filename.replace('.png', ''))
                timestamps.append(timestamp)
            except ValueError:
                continue
        
        return sorted(timestamps)
    
    def _load_image(self, timestamp: int, image_type: str) -> Optional[np.ndarray]:
        """加载指定时间戳和类型的图像"""
        dir_map = {
            'left_rectify': self.left_rectify_dir,
            'left_difference': self.left_difference_dir,
            'right_rectify': self.right_rectify_dir,
            'right_difference': self.right_difference_dir,
        }
        
        if image_type not in dir_map:
            return None
        
        image_path = os.path.join(dir_map[image_type], f"{timestamp}.png")
        if not os.path.exists(image_path):
            return None
        
        try:
            # 使用PIL加载图像，保持原始三通道格式
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')  # 只转换RGBA到RGB
            elif img.mode == 'L':
                img = img.convert('RGB')  # 灰度转RGB
            # 如果已经是RGB，保持不变
            
            img_array = np.array(img)
            
            # 确保数据类型为uint8
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            return img_array
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None
    
    def visualize_gripper_log(self):
        """可视化夹爪日志曲线"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(f'Gripper Control Log - {self.sample_name}', fontsize=14)
        
        # Plot: command vs feedback comparison
        ax.plot(self.gripper_data['time_s'], self.gripper_data['cmd_mm'], 
                'b-', label='Command (mm)', linewidth=2)
        
        # Filter out invalid feedback values
        valid_fb = self.gripper_data['fb_mm'].notna()
        if valid_fb.any():
            ax.plot(self.gripper_data.loc[valid_fb, 'time_s'], 
                    self.gripper_data.loc[valid_fb, 'fb_mm'], 
                    'r-', label='Feedback (mm)', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gripper Opening (mm)')
        ax.set_title('Gripper Position Control')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_object_points(self):
        """可视化带接触标记的物体点云"""
        if not HAS_OPEN3D or self.object_points is None:
            print("跳过3D点云可视化（缺少open3d或点云数据）")
            return None
        
        # 分离坐标和接触标记
        points_3d = self.object_points[:, :3]  # x, y, z
        contact_labels = self.object_points[:, 3].astype(int)  # 接触标记
        
        # 创建颜色映射
        colors = np.zeros((len(points_3d), 3))
        
        # 0: 未接触 - 蓝色
        mask_no_contact = (contact_labels == 0)
        colors[mask_no_contact] = [0.0, 0.0, 1.0]  # 蓝色
        
        # 1: 左夹爪接触 - 绿色
        mask_left_contact = (contact_labels == 1)
        colors[mask_left_contact] = [0.0, 1.0, 0.0]  # 绿色
        
        # 2: 右夹爪接触 - 红色
        mask_right_contact = (contact_labels == 2)
        colors[mask_right_contact] = [1.0, 0.0, 0.0]  # 红色
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 统计信息
        total_points = len(points_3d)
        no_contact_count = np.sum(mask_no_contact)
        left_contact_count = np.sum(mask_left_contact)
        right_contact_count = np.sum(mask_right_contact)
        
        print(f"[点云统计] 总点数: {total_points}")
        print(f"  - 未接触: {no_contact_count} ({no_contact_count/total_points*100:.1f}%)")
        print(f"  - 左夹爪接触: {left_contact_count} ({left_contact_count/total_points*100:.1f}%)")
        print(f"  - 右夹爪接触: {right_contact_count} ({right_contact_count/total_points*100:.1f}%)")
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        
        # Visualize
        print("正在打开3D点云窗口...")
        print("控制说明:")
        print("  - 鼠标左键拖拽: 旋转")
        print("  - 鼠标右键拖拽: 平移")
        print("  - 滚轮: 缩放")
        print("  - 按ESC或关闭窗口退出")
        
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name=f"Object Point Cloud - {self.sample_name}",
            width=1200,
            height=800,
            point_show_normal=False
        )
        
        return pcd
    
    def create_tactile_video_animation(self):
        """创建触觉图像的动画播放"""
        if len(self.image_timestamps) == 0:
            print("没有找到触觉图像，跳过动画创建")
            return None, None
        
        # 创建图形布局 - 包含四个图像和夹爪控制曲线
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.2)
        
        # 四个图像子图
        ax_left_rect = fig.add_subplot(gs[0, 0])
        ax_right_rect = fig.add_subplot(gs[0, 1])
        ax_left_diff = fig.add_subplot(gs[1, 0])
        ax_right_diff = fig.add_subplot(gs[1, 1])
        
        # 夹爪日志子图
        ax_gripper = fig.add_subplot(gs[2, :])
        
        # 设置标题
        fig.suptitle(f'Tactile Sample Data Playback - {self.sample_name}', fontsize=16)
        
        ax_left_rect.set_title('Left Rectify')
        ax_right_rect.set_title('Right Rectify')
        ax_left_diff.set_title('Left Difference')
        ax_right_diff.set_title('Right Difference')
        ax_gripper.set_title('Gripper Position')
        
        # 初始化图像显示 - 使用700*400尺寸的占位符
        placeholder = np.zeros((700, 400, 3), dtype=np.uint8)
        
        # 直接显示RGB图像，不使用颜色映射
        im_left_rect = ax_left_rect.imshow(placeholder, aspect='equal')
        im_right_rect = ax_right_rect.imshow(placeholder, aspect='equal')
        im_left_diff = ax_left_diff.imshow(placeholder, aspect='equal')
        im_right_diff = ax_right_diff.imshow(placeholder, aspect='equal')
        
        # 移除坐标轴
        for ax in [ax_left_rect, ax_right_rect, ax_left_diff, ax_right_diff]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 初始化夹爪曲线
        ax_gripper.plot(self.gripper_data['time_s'], self.gripper_data['cmd_mm'], 
                       'b-', label='Command (mm)', alpha=0.7, linewidth=1)
        
        valid_fb = self.gripper_data['fb_mm'].notna()
        if valid_fb.any():
            ax_gripper.plot(self.gripper_data.loc[valid_fb, 'time_s'], 
                           self.gripper_data.loc[valid_fb, 'fb_mm'], 
                           'r-', label='Feedback (mm)', alpha=0.7, linewidth=1)
        
        # 当前时间指示线
        time_line = ax_gripper.axvline(x=0, color='orange', linewidth=3, label='Current Time')
        
        ax_gripper.set_xlabel('Time (s)')
        ax_gripper.set_ylabel('Gripper Opening (mm)')
        ax_gripper.legend()
        ax_gripper.grid(True, alpha=0.3)
        
        # 时间文本
        time_text = fig.text(0.02, 0.95, '', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame_idx):
            """动画更新函数"""
            if frame_idx >= len(self.image_timestamps):
                return []
            
            timestamp = self.image_timestamps[frame_idx]
            time_s = timestamp / 1000.0
            
            # 更新时间显示
            time_text.set_text(f'Time: {time_s:.2f}s (Frame {frame_idx+1}/{len(self.image_timestamps)})')
            
            # 加载并显示四张图像
            images = {
                'left_rectify': self._load_image(timestamp, 'left_rectify'),
                'right_rectify': self._load_image(timestamp, 'right_rectify'),
                'left_difference': self._load_image(timestamp, 'left_difference'),
                'right_difference': self._load_image(timestamp, 'right_difference'),
            }
            
            # 更新图像 - 直接显示RGB图像
            if images['left_rectify'] is not None:
                im_left_rect.set_array(images['left_rectify'])
            if images['right_rectify'] is not None:
                im_right_rect.set_array(images['right_rectify'])
            if images['left_difference'] is not None:
                im_left_diff.set_array(images['left_difference'])
            if images['right_difference'] is not None:
                im_right_diff.set_array(images['right_difference'])
            
            # 更新时间线
            time_line.set_xdata([time_s, time_s])
            
            return [im_left_rect, im_right_rect, im_left_diff, im_right_diff, time_line, time_text]
        
        # 创建动画
        print(f"创建动画，共 {len(self.image_timestamps)} 帧...")
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.image_timestamps),
            interval=100, blit=False, repeat=True
        )
        
        return anim, fig
    
    def run_full_visualization(self):
        """运行完整的可视化流程"""
        print(f"\n=== 开始可视化采样数据: {self.sample_name} ===\n")
        
        # 1. Create tactile video animation first
        print("1. 创建触觉图像动画...")
        anim, video_fig = self.create_tactile_video_animation()
        if anim is not None:
            plt.show(block=False)  # 非阻塞显示
        
        # 2. Display 3D point cloud (if available) - 独立窗口
        if HAS_OPEN3D and self.object_points is not None:
            print("2. 显示3D点云（独立窗口）...")
            # 使用线程来独立显示点云，避免阻塞
            import threading
            def show_pointcloud():
                self.visualize_object_points()
            
            pointcloud_thread = threading.Thread(target=show_pointcloud)
            pointcloud_thread.daemon = True
            pointcloud_thread.start()
        
        # 3. Keep matplotlib windows open
        print("\n=== 可视化完成 ===")
        print("控制说明:")
        print("  - 点云窗口和动态图窗口同时显示")
        print("  - 动画会自动循环播放")
        print("  - 关闭所有matplotlib窗口结束程序")
        
        try:
            plt.show()  # Block until all windows are closed
        except KeyboardInterrupt:
            print("\n用户中断，正在退出...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Tactile Sample Data Visualization Tool')
    parser.add_argument('sample_dir', help='Sample data directory path')
    parser.add_argument('--gripper-only', action='store_true', help='Display gripper log only')
    parser.add_argument('--points-only', action='store_true', help='Display point cloud only')
    parser.add_argument('--animation-only', action='store_true', help='Display tactile animation only')
    
    args = parser.parse_args()
    
    # Verify directory exists
    if not os.path.isdir(args.sample_dir):
        print(f"错误: 目录不存在: {args.sample_dir}")
        return 1
    
    try:
        # Create visualizer
        visualizer = TactileSampleVisualizer(args.sample_dir)
        
        if args.gripper_only:
            # Display gripper log only
            print("仅显示夹爪日志...")
            fig = visualizer.visualize_gripper_log()
            plt.show()
        elif args.points_only:
            # Display point cloud only
            print("仅显示点云...")
            visualizer.visualize_object_points()
        elif args.animation_only:
            # Display tactile animation only
            print("仅显示触觉动画...")
            anim, fig = visualizer.create_tactile_video_animation()
            if anim is not None:
                plt.show()
        else:
            # Full visualization
            visualizer.run_full_visualization()
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
