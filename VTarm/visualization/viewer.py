import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from typing import Optional, Tuple, List


class Viewer:
    """
    通用可视化器：支持3D点云与2D图像的统一显示与更新。

    - 3D 使用 Open3D
    - 2D 使用 Matplotlib + Tkinter
    - 提供统一的 update()/update2d()/update3d_pointcloud() 接口
    - 支持设置窗口大小与2D面板布局
    """

    def __init__(self,
                 window_title: str = "可视化",
                 window_width: int = 1600,
                 window_height: int = 900,
                 num_2d_panels: int = 2,
                 o3d_ratio: float = 0.65) -> None:
        self.window_title = window_title
        self.window_width = window_width
        self.window_height = window_height
        self.num_2d_panels = max(0, int(num_2d_panels))
        self.o3d_ratio = float(np.clip(o3d_ratio, 0.3, 0.9))
        # 3D 窗口懒加载：首次使用3D接口时创建

        # 3D
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.geometries: dict = {}
        self.view_control = None

        # 2D（右侧）
        self.depth_window: Optional[tk.Tk] = None
        self.depth_fig = None
        self.depth_axes: Optional[List] = None
        self.depth_canvas = None

        self.running = True

    # -------------------- 基础设置 --------------------
    def setup(self) -> bool:
        try:
            # 初始不创建任何窗口；
            # - 3D 在首次使用3D接口或显式 open_3d() 时创建
            # - 2D 在显式 open_2d() 时创建
            return True
        except Exception as e:
            print(f"可视化设置失败: {e}")
            return False

    def _setup_2d_window(self) -> None:
        try:
            o3d_width = int(self.window_width * self.o3d_ratio)
            depth_width = max(300, self.window_width - o3d_width)

            self.depth_window = tk.Tk()
            self.depth_window.title("2D Panels")
            self.depth_window.geometry(f"{depth_width}x{self.window_height}+{o3d_width}+0")
            self.depth_window.resizable(False, False)

            # 支持2x2布局：当有4个面板时使用2x2，否则使用垂直排列
            if self.num_2d_panels == 4:
                rows, cols = 2, 2
                self.depth_fig, self.depth_axes = plt.subplots(rows, cols, figsize=(depth_width/100, self.window_height/100))
                # 将2D数组展平为1D列表
                self.depth_axes = self.depth_axes.flatten()
            else:
                rows = self.num_2d_panels
                if rows <= 0:
                    rows = 1
                self.depth_fig, self.depth_axes = plt.subplots(rows, 1, figsize=(depth_width/100, self.window_height/100))
                if rows == 1:
                    self.depth_axes = [self.depth_axes]

            for i in range(self.num_2d_panels):
                empty = np.zeros((50, 50))
                self.depth_axes[i].imshow(empty, cmap='gray', origin='lower')
                self.depth_axes[i].set_title(f'Panel {i}', fontsize=10)
                self.depth_axes[i].set_xlabel('Width')
                self.depth_axes[i].set_ylabel('Height')

            plt.tight_layout()
            self.depth_canvas = FigureCanvasTkAgg(self.depth_fig, self.depth_window)
            self.depth_canvas.draw()
            self.depth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            def on_close():
                self.running = False
                self.depth_window.destroy()
            self.depth_window.protocol("WM_DELETE_WINDOW", on_close)
        except Exception as e:
            print(f"2D窗口设置失败: {e}")
            self.depth_window = None
            self.depth_axes = None

    def open_2d(self, *, num_2d_panels: int = None) -> None:
        """显式打开/重建 2D 窗口，可选调整面板数。"""
        try:
            if num_2d_panels is not None and num_2d_panels != self.num_2d_panels:
                self.num_2d_panels = max(0, int(num_2d_panels))
                self.close_2d()
            if self.depth_window is None:
                self._setup_2d_window()
        except Exception as e:
            print(f"打开2D窗口失败: {e}")

    def close_2d(self) -> None:
        """关闭 2D 窗口并清理资源。"""
        if self.depth_window:
            try:
                self.depth_window.destroy()
            except Exception as e:
                print(f"关闭2D窗口出错: {e}")
            self.depth_window = None
        if self.depth_fig is not None:
            try:
                plt.close(self.depth_fig)
            except Exception:
                pass
        self.depth_fig = None
        self.depth_axes = None
        self.depth_canvas = None

    def _register_callbacks(self) -> None:
        def on_quit(vis_):
            self.running = False
            return False
        def reset_view(vis_):
            if self.view_control:
                self.view_control.set_front([0, 0, -1])
                self.view_control.set_up([0, -1, 0])
                self.view_control.set_lookat([0, 0, 0])
                self.view_control.set_zoom(0.8)
            return False
        self.vis.register_key_callback(ord('Q'), on_quit)
        self.vis.register_key_callback(ord('R'), reset_view)

    def _ensure_3d_window(self) -> None:
        if self.vis is not None:
            return
        try:
            o3d_width = int(self.window_width * self.o3d_ratio)
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(
                window_name=self.window_title,
                width=o3d_width,
                height=self.window_height,
                left=0,
                top=0
            )
            try:
                opt = self.vis.get_render_option()
                opt.point_size = 4.0
                opt.background_color = np.array([0.1, 0.1, 0.1])
            except Exception:
                pass
            self.view_control = self.vis.get_view_control()
            self._register_callbacks()
        except Exception as e:
            print(f"创建3D窗口失败: {e}")

    def open_3d(self) -> None:
        """显式打开 3D 窗口。"""
        self._ensure_3d_window()

    def close_3d(self) -> None:
        """关闭 3D 窗口并清理几何对象。"""
        if self.vis:
            try:
                self.vis.destroy_window()
            except Exception as e:
                print(f"关闭3D窗口出错: {e}")
            self.vis = None
        self.geometries.clear()
        self.view_control = None

    # -------------------- API：布局与窗口 --------------------
    def set_window_size(self, width: int, height: int) -> None:
        self.window_width = int(width)
        self.window_height = int(height)
        # 需要重建窗口时，调用者应当先 close() 再 setup()

    def set_layout(self, *, num_2d_panels: int = None, o3d_ratio: float = None) -> None:
        if num_2d_panels is not None:
            self.num_2d_panels = max(0, int(num_2d_panels))
        if o3d_ratio is not None:
            self.o3d_ratio = float(np.clip(o3d_ratio, 0.3, 0.9))
        # 需要重建窗口时，调用者应当先 close() 再 setup()

    # -------------------- API：3D --------------------
    def add_coordinate_frame(self, name: str, size: float = 50.0) -> None:
        self._ensure_3d_window()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._add_geometry(name, axis)

    def add_coordinate_frame_with_pose(self, name: str, size: float, position: Tuple[float, float, float], rotation_matrix: Optional[np.ndarray]) -> None:
        self._ensure_3d_window()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        pos = list(position) if position is not None else [0.0, 0.0, 0.0]
        if pos != [0.0, 0.0, 0.0]:
            axis.translate(pos)
        if rotation_matrix is not None:
            axis.rotate(rotation_matrix, center=pos)
        self._add_geometry(name, axis)

    def add_line_set(self, name: str, points: np.ndarray, lines: np.ndarray, colors: np.ndarray = None) -> None:
        self._ensure_3d_window()
        if name in self.geometries and isinstance(self.geometries[name], o3d.geometry.LineSet):
            line_set = self.geometries[name]
            line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
            if colors is not None:
                line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            self.vis.update_geometry(line_set)
        else:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
            if colors is not None:
                line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            self._add_geometry(name, line_set)

    def update3d_pointcloud(self, name: str, points: np.ndarray, colors: np.ndarray = None) -> None:
        self._ensure_3d_window()
        if points.size == 0 or points.shape[0] == 0:
            if name in self.geometries:
                self._remove_geometry(name)
            return
        if name in self.geometries and isinstance(self.geometries[name], o3d.geometry.PointCloud):
            pcd = self.geometries[name]
        else:
            pcd = o3d.geometry.PointCloud()
            self._add_geometry(name, pcd)
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and colors.size > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        self.vis.update_geometry(pcd)

    # -------------------- API：2D --------------------
    def update2d(self, panel_index: int, image: np.ndarray, *, cmap: str = 'gray', title: Optional[str] = None) -> None:
        if self.depth_window is None or self.depth_axes is None:
            return
        if panel_index < 0 or panel_index >= len(self.depth_axes):
            return
        ax = self.depth_axes[panel_index]
        ax.clear()
        if image is not None and image.size > 0:
            # 直接使用原始图像，不做动态范围调整（保持全灰色效果）
            ax.imshow(image, cmap=cmap, origin='lower')
            h, w = image.shape[:2]
            title_text = title if title is not None else f'Panel {panel_index} ({h}x{w})'
            ax.set_title(title_text, fontsize=10)
        else:
            empty = np.zeros((50, 50))
            ax.imshow(empty, cmap=cmap, origin='lower')
            ax.set_title(f'Panel {panel_index} (No Data)', fontsize=10)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

    def update(self, sleep_time: float = 0.01) -> bool:
        # 3D 事件（如果启用）
        if self.vis is not None:
            if not self.vis.poll_events():
                self.running = False
                # 继续让2D窗口可运行，直到2D窗口关闭
            else:
                self.vis.update_renderer()
        if self.depth_canvas:
            try:
                self.depth_canvas.draw()
            except Exception:
                pass
        if self.depth_window:
            try:
                self.depth_window.update_idletasks()
            except Exception:
                pass
        time.sleep(sleep_time)
        return self.running

    def is_running(self) -> bool:
        """兼容旧接口，返回当前可视化是否仍在运行。"""
        return bool(self.running)

    # -------------------- 内部辅助 --------------------
    def _add_geometry(self, name: str, geometry) -> None:
        if name in self.geometries:
            self._remove_geometry(name)
        self.geometries[name] = geometry
        self.vis.add_geometry(geometry)

    def _remove_geometry(self, name: str) -> None:
        if name in self.geometries:
            try:
                self.vis.remove_geometry(self.geometries[name])
            except Exception:
                pass
            del self.geometries[name]

    @staticmethod
    def _normalize_to_float(img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        mn = float(np.min(img_f))
        mx = float(np.max(img_f))
        if mx - mn < 1e-6:
            return np.zeros_like(img_f, dtype=np.float32)
        return (img_f - mn) / (mx - mn)

    # -------------------- 关闭 --------------------
    def close(self) -> None:
        # 先停止运行标志
        self.running = False
        
        # 关闭2D窗口
        if self.depth_window:
            try:
                self.depth_window.quit()  # 先退出事件循环
                self.depth_window.destroy()
            except Exception as e:
                print(f"关闭2D窗口出错: {e}")
            finally:
                self.depth_window = None
        
        # 关闭3D窗口
        if self.vis:
            try:
                # 先清理所有几何体
                for name in list(self.geometries.keys()):
                    try:
                        self.vis.remove_geometry(self.geometries[name])
                    except Exception:
                        pass
                self.geometries.clear()
                
                # 然后销毁窗口
                self.vis.destroy_window()
            except Exception as e:
                print(f"关闭3D窗口出错: {e}")
            finally:
                self.vis = None
        
        # 清理其他资源
        self.view_control = None
        if self.depth_fig is not None:
            try:
                plt.close(self.depth_fig)
            except Exception:
                pass
        self.depth_fig = None
        self.depth_axes = None
        self.depth_canvas = None


