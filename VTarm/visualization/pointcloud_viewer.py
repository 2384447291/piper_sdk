import numpy as np
import open3d as o3d
import time
from typing import Optional, Tuple


class PointCloudViewer:
    """
    通用点云可视化类，封装Open3D可视化功能
    支持动态添加和移除任意几何对象（点云、坐标轴、线框等）
    """
    
    def __init__(self, 
                 window_title: str = "点云可视化", 
                 window_width: int = 1200, 
                 window_height: int = 800):
        """
        初始化点云可视化器
        
        Args:
            window_title: 窗口标题
            window_width: 窗口宽度
            window_height: 窗口高度
        """
        self.window_title = window_title
        self.window_width = window_width
        self.window_height = window_height
        
        # 可视化组件
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.geometries: dict = {}  # 存储所有几何对象 {name: geometry}
        
        # 控制状态
        self.running = True
        self.view_control = None
        
    def setup(self) -> bool:
        """
        设置可视化环境
            
        Returns:
            bool: 设置是否成功
        """
        try:
            # 创建可视化窗口
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(
                window_name=self.window_title, 
                width=self.window_width, 
                height=self.window_height
            )
            
            # 设置渲染选项
            try:
                opt = self.vis.get_render_option()
                opt.point_size = 2.0
                opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            except Exception:
                pass
            
            # 获取视角控制器
            self.view_control = self.vis.get_view_control()
            
            # 注册键盘回调
            self._register_callbacks()
            
            # 显示控制说明
            self._print_controls()
            
            return True
            
        except Exception as e:
            print(f"可视化设置失败: {e}")
            return False
    
    def _register_callbacks(self):
        """注册键盘回调函数"""
        def on_quit(vis_):
            self.running = False
            return False
        
        def reset_view(vis_):
            """重置视角"""
            if self.view_control:
                self.view_control.set_front([0, 0, -1])
                self.view_control.set_up([0, -1, 0])
                self.view_control.set_lookat([0, 0, 0])
                self.view_control.set_zoom(0.8)
            return False
        
        # 只注册必要的回调，让Open3D处理鼠标和滚轮
        self.vis.register_key_callback(ord('Q'), on_quit)
        self.vis.register_key_callback(ord('R'), reset_view)
    
    def _print_controls(self):
        """打印控制说明"""
        print("\n=== 控制说明 ===")
        print("Q: 退出程序")
        print("R: 重置视角")
        print("鼠标左键拖动: 旋转视角")
        print("鼠标右键拖动: 平移视角")
        print("鼠标滚轮: 缩放视角")
        print("==================\n")
    
    def add_geometry(self, name: str, geometry):
        """
        添加几何对象到可视化器
        
        Args:
            name: 几何对象名称（用于后续更新或移除）
            geometry: Open3D几何对象（点云、线框、坐标轴等）
        """
        if self.vis is None:
            print("请先调用setup()初始化可视化器")
            return False
        
        # 如果已存在同名对象，先移除
        if name in self.geometries:
            self.remove_geometry(name)
        
        self.geometries[name] = geometry
        self.vis.add_geometry(geometry)
        return True
    
    def update_geometry(self, name: str, geometry=None):
        """
        更新几何对象
        
        Args:
            name: 几何对象名称
            geometry: 新的几何对象（如果为None，则更新现有对象）
        """
        if name not in self.geometries:
            print(f"几何对象 '{name}' 不存在")
            return False
        
        if geometry is not None:
            # 替换几何对象
            self.remove_geometry(name)
            self.add_geometry(name, geometry)
        else:
            # 更新现有几何对象
            self.vis.update_geometry(self.geometries[name])
        return True
    
    def remove_geometry(self, name: str):
        """
        移除几何对象
        
        Args:
            name: 几何对象名称
        """
        if name in self.geometries:
            self.vis.remove_geometry(self.geometries[name])
            del self.geometries[name]
            return True
        return False
    
    def update_pointcloud(self, name: str, points: np.ndarray, colors: np.ndarray = None):
        """
        更新或创建点云
        
        Args:
            name: 点云名称
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3)，可选
        """
        if points.size == 0 or points.shape[0] == 0:
            # 如果点云为空，移除该点云
            if name in self.geometries:
                self.remove_geometry(name)
            return
        
        # 创建或更新点云
        if name in self.geometries and isinstance(self.geometries[name], o3d.geometry.PointCloud):
            # 更新现有点云
            pcd = self.geometries[name]
        else:
            # 创建新点云
            pcd = o3d.geometry.PointCloud()
            self.add_geometry(name, pcd)
        
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and colors.size > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
        self.vis.update_geometry(pcd)
    
    def add_coordinate_frame(self, name: str, size: float = 50.0, origin: Tuple[float, float, float] = (0, 0, 0)):
        """
        添加坐标轴
        
        Args:
            name: 坐标轴名称
            size: 坐标轴大小
            origin: 坐标轴原点位置 (x, y, z)
        """
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        if origin != (0, 0, 0):
            axis.translate(list(origin))
        self.add_geometry(name, axis)
    
    def add_coordinate_frame_with_pose(self,
                                       name: str,
                                       size: float = 50.0,
                                       position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                       rotation_matrix: Optional[np.ndarray] = None):
        """
        添加带位姿的坐标轴（先平移至 position，再绕该点旋转）。
        
        Args:
            name: 坐标轴名称
            size: 坐标轴大小
            position: 平移位置 (x, y, z)
            rotation_matrix: 3x3 旋转矩阵（可选）；若提供，则以 position 为中心进行旋转
        """
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        pos = list(position) if position is not None else [0.0, 0.0, 0.0]
        if pos != [0.0, 0.0, 0.0]:
            axis.translate(pos)
        if rotation_matrix is not None:
            axis.rotate(rotation_matrix, center=pos)
        self.add_geometry(name, axis)
    
    def add_line_set(self, name: str, points: np.ndarray, lines: np.ndarray, colors: np.ndarray = None):
        """
        添加或更新线框
        
        Args:
            name: 线框名称
            points: 顶点坐标 (N, 3)
            lines: 线段连接 (M, 2)
            colors: 线段颜色 (M, 3)，可选
        """
        # 如果已存在同名线框，更新它而不是重新创建
        if name in self.geometries and isinstance(self.geometries[name], o3d.geometry.LineSet):
            line_set = self.geometries[name]
            line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
            if colors is not None:
                line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            self.vis.update_geometry(line_set)
        else:
            # 创建新线框
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
            if colors is not None:
                line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            self.add_geometry(name, line_set)
    
    def poll_events(self) -> bool:
        """
        处理窗口事件
        
        Returns:
            bool: 是否继续运行
        """
        if self.vis is None:
            return False
        # poll_events()处理所有事件包括鼠标交互
        if not self.vis.poll_events():
            self.running = False
            return False
        return self.running
    
    def update_renderer(self):
        """更新渲染器"""
        if self.vis:
            self.vis.update_renderer()
    
    def close(self):
        """关闭可视化窗口并释放资源"""
        if self.vis:
            try:
                self.vis.destroy_window()
            except Exception as e:
                print(f"关闭可视化窗口时出错: {e}")
            self.vis = None
        
        # 清理几何对象引用
        self.geometries.clear()
        self.view_control = None
    
    def is_running(self) -> bool:
        """检查是否仍在运行"""
        return self.running
    
    def run_update_cycle(self, sleep_time: float = 0.01):
        """
        运行一次更新循环
        
        Args:
            sleep_time: 循环间隔时间(秒)
        """
        if not self.poll_events():
            return False
        self.update_renderer()
        time.sleep(sleep_time)  # 减少睡眠时间以提高响应性
        return True