"""
线程间共享数据类
用于多线程架构中的数据同步和状态管理
"""
import threading
import numpy as np
from typing import Optional
from enum import Enum


class FlowState(Enum):
    """流程状态枚举"""
    CONNECT_ENABLE = 1
    MOVE_TO_ZERO = 2
    START_TRACKING = 3
    WAIT_TRACK_END = 4
    VISUALIZATION = 5
    GRIPPER_CONTROL = 6
    SHUTDOWN = 7


class SharedData:
    """线程间共享数据，带锁保护"""
    
    def __init__(self):
        # 线程锁
        self.lock = threading.Lock()
        
        # 控制相关
        self.gripper_mm = 70.0
        self.is_sampling = False
        self.sample_dir = None
        self.left_rectify_dir = None
        self.left_difference_dir = None
        self.right_rectify_dir = None
        self.right_difference_dir = None
        self.sample_start_time = 0.0
        
        # 触觉数据
        self.left_rectify = None
        self.right_rectify = None
        self.left_difference = None
        self.right_difference = None
        
        
        # 控制标志
        self.shutdown = False
        self.restart_requested = False  # 请求全局重启
        self.restarting = False         # 重启流程进行中
        self.viewer_close_requested = False  # 请求主线程关闭viewer（2D窗口）
        self.state = FlowState.CONNECT_ENABLE
        self.viewer = None
        self.tactile_left = None
        self.tactile_right = None
        
        # 可视化数据
        self.avg_pose = None
        self.model_points = None
        self.left_contact_points = None
        self.right_contact_points = None
        self.left_contact_colors = None
        self.right_contact_colors = None
        self.transformed_model = None
        self.model_colors = None
        self.object_points_with_contact = None
    
    def _reset_visualization_data_unlocked(self):
        """重置可视化相关数据（不加锁）"""
        self.avg_pose = None
        self.model_points = None
        self.left_contact_points = None
        self.right_contact_points = None
        self.left_contact_colors = None
        self.right_contact_colors = None
        self.transformed_model = None
        self.model_colors = None
        self.object_points_with_contact = None
    
    def _reset_tactile_data_unlocked(self):
        """重置触觉相关数据（不加锁）"""
        self.left_rectify = None
        self.right_rectify = None
        self.left_difference = None
        self.right_difference = None
    
    def reset_visualization_data(self):
        """重置可视化相关数据"""
        with self.lock:
            self._reset_visualization_data_unlocked()
    
    def reset_tactile_data(self):
        """重置触觉相关数据"""
        with self.lock:
            self._reset_tactile_data_unlocked()
    
    def close_viewer(self):
        """安全关闭可视化窗口（无锁，最高优先级）"""
        print("[SharedData] 开始关闭可视化窗口...")
        try:
            # 无锁获取viewer引用
            viewer = self.viewer
            if viewer is not None:
                print("[SharedData] 正在停止可视化循环...")
                viewer.running = False  # 立即停止更新循环
                
                print("[SharedData] 正在关闭可视化窗口...")
                viewer.close()  # 直接关闭，不使用超时
                print("[SharedData] 可视化窗口已关闭")
                
                # 最后清空引用（无锁）
                self.viewer = None
        except Exception as e:
            print(f"[SharedData] 关闭可视化窗口失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使失败也清空引用
            self.viewer = None
    
    def close_tactile_sensors(self):
        """安全关闭触觉传感器（无锁，最高优先级）"""
        print("[SharedData] 开始关闭触觉传感器...")
        try:
            # 无锁获取传感器引用
            tactile_left = self.tactile_left
            tactile_right = self.tactile_right
            
            if tactile_left is not None:
                try:
                    print("[SharedData] 正在关闭左侧触觉传感器...")
                    tactile_left.close()
                    print("[SharedData] 左侧触觉传感器已关闭")
                except Exception as e:
                    print(f"[SharedData] 关闭左侧触觉传感器失败: {e}")
            
            if tactile_right is not None:
                try:
                    print("[SharedData] 正在关闭右侧触觉传感器...")
                    tactile_right.close()
                    print("[SharedData] 右侧触觉传感器已关闭")
                except Exception as e:
                    print(f"[SharedData] 关闭右侧触觉传感器失败: {e}")
            
            # 最后清空引用（无锁）
            self.tactile_left = None
            self.tactile_right = None
                    
        except Exception as e:
            print(f"[SharedData] 关闭触觉传感器时出错: {e}")
            import traceback
            traceback.print_exc()
            # 即使失败也清空引用
            self.tactile_left = None
            self.tactile_right = None
    
    def cleanup_all(self):
        """清理所有资源"""
        print("[SharedData] 开始清理所有资源...")
        try:
            # 分步清理，避免长时间持有锁
            print("[SharedData] 正在关闭可视化窗口...")
            self.close_viewer()
            print("[SharedData] 正在关闭触觉传感器...")
            self.close_tactile_sensors()
            
            with self.lock:
                print("[SharedData] 正在重置数据...")
                self._reset_visualization_data_unlocked()
                self._reset_tactile_data_unlocked()
                self.is_sampling = False
                self.sample_dir = None
                self.left_rectify_dir = None
                self.left_difference_dir = None
                self.right_rectify_dir = None
                self.right_difference_dir = None
            print("[SharedData] 所有资源清理完成")
        except Exception as e:
            print(f"[SharedData] 清理资源时出错: {e}")
            import traceback
            traceback.print_exc()
