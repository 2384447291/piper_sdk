import sys
import numpy as np
import open3d as o3d 
from xensesdk import Sensor
import time
from typing import Optional

# --- 关键配置：传感器的物理尺寸 (由客服确认的精确值) ---
SENSOR_WIDTH_MM = 19.4  # 传感器的物理宽度为19.4mm (对应400像素)
SENSOR_HEIGHT_MM = 30.8 # 传感器的物理高度为30.8mm (对应700像素)

# --- 可视化：统一颜色（用于平坦/无变化时回退） ---
UNIFORM_COLOR = np.array([0.1, 0.6, 1.0], dtype=np.float64)  # RGB in [0,1]

# 注：不再使用全局阈值常量，阈值由类 `TactileSensorViewer` 的实例变量控制


def scaled_depth_to_points_and_z(depth_map, z_scale=1.0):
    """
    将深度图转换为点云，并返回展平后的 Z（毫米）。
    """
    h, w = depth_map.shape
    v, u = np.indices((h, w))

    x = (u / w) * SENSOR_WIDTH_MM
    y = (v / h) * SENSOR_HEIGHT_MM

    safe_depth = np.nan_to_num(depth_map, nan=0.0)
    z = safe_depth * z_scale

    points = np.dstack((x, y, z)).reshape(-1, 3)
    z_flat = z.reshape(-1)
    return points, z_flat


def depth_to_filtered_point_cloud(
    depth_map: np.ndarray,
    *,
    z_scale: float = 1.0,
    z_threshold_mm: float,
    return_z: bool = False,
) -> np.ndarray:
    """
    将深度图直接转换为点云并按阈值过滤，仅返回 Z 大于阈值的点。

    参数：
    - depth_map：输入深度图（单位与传感器一致）
    - z_scale：Z 轴缩放系数
    - z_threshold_mm：Z 阈值（毫米）
    - return_z：True 则同时返回过滤后的 Z 向量

    返回：
    - 若 return_z=False：返回 (N,3) 的点云 ndarray（float64）
    - 若 return_z=True：返回 (points, z_flat_filtered)
    """
    points, z_flat = scaled_depth_to_points_and_z(depth_map, z_scale=z_scale)
    mask = z_flat > float(z_threshold_mm)
    filtered_points = points[mask].astype(np.float64)
    if return_z:
        return filtered_points, z_flat[mask]
    return filtered_points


def colors_from_height(z_flat: np.ndarray, max_mm: float = 1.0) -> np.ndarray:
    """
    根据绝对高度（毫米）生成颜色：将 z 线性映射到 [0,1]，其中 0mm -> 蓝色，max_mm -> 红色。
    超出范围的值进行裁剪；非有限值位置回退为统一颜色。
    返回形状 (N,3)，dtype=float64。
    """
    if z_flat.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = np.asarray(z_flat, dtype=np.float64)
    # 直接按 0~1mm 绝对范围归一化
    max_mm = float(max(1e-9, max_mm))
    t = np.clip(z / max_mm, 0.0, 1.0)

    # 简单蓝-红渐变：蓝(低)->青->黄->红(高)
    r = t
    g = 0.5 * (1.0 - np.abs(2.0 * t - 1.0)) + 0.1  # 中间偏亮
    b = 1.0 - t
    colors = np.stack([r, g, b], axis=1)
    colors = np.clip(colors, 0.0, 1.0)

    # 非有限值位置使用统一颜色
    valid = np.isfinite(z)
    if not np.all(valid):
        colors[~valid] = UNIFORM_COLOR

    return colors.astype(np.float64)


class TactileSensorViewer:
    """触觉传感器点云可视化（类封装）。

    - 负责传感器连接、首帧等待、点云过滤、着色、Open3D 可视化窗口管理与实时刷新。
    - 仅依赖本文件的工具函数 `scaled_depth_to_points_and_z` 与 `colors_from_height`。
    """

    def __init__(
        self,
        sensor_serial: str = "OG000229",
        *,
        z_threshold_mm: float = 0.02,
        color_max_mm: float = 1.0,
        point_size: float = 3.0,
        window_name: str = "Raw Dense Point Cloud (Height-colored)",
        z_scale_factor: float = 1.0,
    ) -> None:
        self.sensor_serial = sensor_serial
        self.z_threshold_mm = float(z_threshold_mm)
        self.color_max_mm = float(color_max_mm)
        self.point_size = float(point_size)
        self.window_name = window_name
        self.z_scale_factor = float(z_scale_factor)

        self.sensor: Optional[Sensor] = None
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.prev_n: int = 0

    # ----------------------------
    # 内部工具
    # ----------------------------
    def _wait_first_frame(self) -> np.ndarray:
        """重置参考，等待首帧有效深度。返回深度图。"""
        print("正在重置传感器参考图像...")
        self.sensor.resetReferenceImage()
        time.sleep(0.5)
        print("重置完成。")

        print("正在等待传感器开启并传输第一帧深度图...")
        depth = None
        for _ in range(100):
            depth = self.sensor.selectSensorInfo(Sensor.OutputType.Depth)
            if depth is not None and np.any(depth > 0):
                break
            time.sleep(0.1)

        if depth is None or not np.any(depth > 0):
            raise RuntimeError("未接收到有效的深度数据")
        print("成功接收到第一帧数据，正在初始化可视化窗口...")
        return depth

    @staticmethod
    def _filter_points(points_3d: np.ndarray, z_flat: np.ndarray, threshold_mm: float) -> tuple:
        """按 Z 阈值进行过滤，返回 (filtered_points_3d, filtered_z_flat)。"""
        mask = z_flat > float(threshold_mm)
        return points_3d[mask], z_flat[mask]

    # ----------------------------
    # 生命周期
    # ----------------------------
    def setup(self, *, visualize: bool = False, wait_first_frame: bool = True) -> None:
        """连接传感器；可选是否初始化可视化。

        参数：
        - visualize：是否创建可视化窗口（默认 False，不创建）
        - wait_first_frame：是否等待首帧有效深度（默认 True）
        """
        self.sensor = Sensor.create(self.sensor_serial)
        depth: Optional[np.ndarray] = None
        if wait_first_frame:
            depth = self._wait_first_frame()

        if not visualize:
            # 仅初始化传感器，不创建可视化
            self.vis = None
            self.pcd = None
            self.prev_n = 0
            return

        # 需要可视化：准备首帧点云（若未等待首帧，则尝试取一帧）
        if depth is None:
            depth = self.sensor.selectSensorInfo(Sensor.OutputType.Depth)
        if depth is None:
            # 没有可用帧也可以创建空窗口
            full_points_3d = np.empty((0, 3), dtype=np.float64)
            filtered_z_flat = np.empty((0,), dtype=np.float64)
            filtered_points_3d = full_points_3d
        else:
            full_points_3d, z_flat = scaled_depth_to_points_and_z(depth, z_scale=self.z_scale_factor)
            filtered_points_3d, filtered_z_flat = self._filter_points(full_points_3d, z_flat, self.z_threshold_mm)

        # 可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name)

        try:
            opt = self.vis.get_render_option()
            opt.point_size = self.point_size
        except Exception:
            pass

        self.pcd = o3d.geometry.PointCloud()
        if filtered_points_3d.shape[0] > 0:
            colors = colors_from_height(filtered_z_flat, self.color_max_mm)
            self.pcd.points = o3d.utility.Vector3dVector(filtered_points_3d.astype(np.float64))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.pcd)
            try:
                self.vis.get_view_control().fit_children()
            except Exception:
                pass
            self.prev_n = filtered_points_3d.shape[0]
        else:
            self.vis.add_geometry(self.pcd)
            self.prev_n = 0

    def update_once(self) -> bool:
        """拉取一帧并更新可视化；窗口关闭时返回 False。"""
        assert self.sensor is not None and self.vis is not None and self.pcd is not None

        new_depth = self.sensor.selectSensorInfo(Sensor.OutputType.Depth)
        if new_depth is not None:
            full_points_3d, z_flat = scaled_depth_to_points_and_z(new_depth, z_scale=self.z_scale_factor)
            filtered_points_3d, filtered_z_flat = self._filter_points(full_points_3d, z_flat, self.z_threshold_mm)
            n = int(filtered_points_3d.shape[0])

            if n > 0:
                colors = colors_from_height(filtered_z_flat, self.color_max_mm)
                if self.prev_n == 0:
                    # 由空 -> 非空，重建并自适应视角
                    try:
                        self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
                    except Exception:
                        pass
                    self.pcd = o3d.geometry.PointCloud()
                    self.pcd.points = o3d.utility.Vector3dVector(filtered_points_3d.astype(np.float64))
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    self.vis.add_geometry(self.pcd)
                    try:
                        self.vis.get_view_control().fit_children()
                    except Exception:
                        pass
                else:
                    self.pcd.points = o3d.utility.Vector3dVector(filtered_points_3d.astype(np.float64))
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    self.vis.update_geometry(self.pcd)
            else:
                # 清空
                empty_pts = np.empty((0, 3), dtype=np.float64)
                empty_cols = np.empty((0, 3), dtype=np.float64)
                self.pcd.points = o3d.utility.Vector3dVector(empty_pts)
                self.pcd.colors = o3d.utility.Vector3dVector(empty_cols)
                self.vis.update_geometry(self.pcd)

            self.prev_n = n

        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def close(self) -> None:
        """释放资源。"""
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass
            self.vis = None
        if self.sensor is not None:
            try:
                self.sensor.release()
            except Exception:
                pass
            self.sensor = None

    def run(self) -> None:
        """运行完整的可视化循环。"""
        try:
            self.setup(visualize=True, wait_first_frame=True)
            keep_running = True
            while keep_running:
                keep_running = self.update_once()
        finally:
            self.close()

    # ----------------------------
    # 无可视化：直接获取点云
    # ----------------------------
    def get_point_cloud(
        self,
        *,
        z_threshold_mm: Optional[float] = None,
        return_z: bool = False,
    ) -> np.ndarray:
        """从传感器抓取一帧并返回阈值过滤后的点云（可选返回 z）。"""
        assert self.sensor is not None, "请先调用 setup(visualize=False) 初始化传感器"
        depth = self.sensor.selectSensorInfo(Sensor.OutputType.Depth)
        if depth is None:
            if return_z:
                return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
            return np.empty((0, 3), dtype=np.float64)
        return depth_to_filtered_point_cloud(
            depth,
            z_scale=self.z_scale_factor,
            z_threshold_mm=self.z_threshold_mm if z_threshold_mm is None else float(z_threshold_mm),
            return_z=return_z,
        )

    def get_mapped_point_cloud(
        self,
        *,
        is_left: bool,
        y_plane_offset: float,
        flip_lr: bool,
        flip_ud: bool,
        return_z: bool = False,
    ) -> np.ndarray:
        """
        获取单帧并将点云从传感器局部坐标系映射到公共坐标系（垂直于Y轴）。

        坐标映射规则：
        - 先可选左右/上下翻转，再以传感器物理尺寸居中（将原点移至传感器中心）
        - 轴映射：Xc <- 图像长边; Zc <- 图像短边; Yc <- 根据左右侧与深度确定
        - 左侧传感器：Y = -y_plane_offset - depth
        - 右侧传感器：Y = +y_plane_offset + depth

        参数：
        - is_left：是否为左侧传感器
        - y_plane_offset：两侧夹爪到传感器面的半距离（mm），即 half_z
        - flip_lr：是否进行左右翻转（以短边/宽度方向）
        - flip_ud：是否进行上下翻转（以长边/高度方向）
        - return_z：是否同时返回对应的 z 向量（用于着色）
        """
        pts, z_flat = self.get_point_cloud(return_z=True)
        if pts.size == 0:
            if return_z:
                return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
            return np.empty((0, 3), dtype=np.float64)

        mapped = pts.copy()
        if flip_lr:
            mapped[:, 0] = SENSOR_WIDTH_MM - mapped[:, 0]
        if flip_ud:
            mapped[:, 1] = SENSOR_HEIGHT_MM - mapped[:, 1]

        # 将局部坐标系原点移至传感器中心
        mapped[:, 0] -= (SENSOR_WIDTH_MM / 2.0)   # 短边（原 x）
        mapped[:, 1] -= (SENSOR_HEIGHT_MM / 2.0)  # 长边（原 y）

        # 轴重映射
        Xc = mapped[:, 1]                  # 长边 -> 全局 X
        depth = mapped[:, 2]               # 深度始终为正
        Zc = mapped[:, 0]                  # 短边 -> 全局 Z
        if is_left:
            Yc = -float(y_plane_offset) - depth
        else:
            Yc = +float(y_plane_offset) + depth
        out = np.stack([Xc, Yc, Zc], axis=1).astype(np.float64)

        if return_z:
            return out, z_flat.astype(np.float64)
        return out


def main():
    sensor_0 = Sensor.create("OG000229")

    print("正在重置传感器参考图像...")
    sensor_0.resetReferenceImage()
    time.sleep(0.5)
    print("重置完成。")

    print("正在等待传感器开启并传输第一帧深度图...")
    depth = None
    for _ in range(100):
        depth = sensor_0.selectSensorInfo(Sensor.OutputType.Depth)
        if depth is not None and np.any(depth > 0):
            break
        time.sleep(0.1)

    if depth is None or not np.any(depth > 0):
        print("错误：未接收到有效的深度数据。")
        sensor_0.release()
        sys.exit()

    print("成功接收到第一帧数据，正在初始化可视化窗口...")

    # 使用真实深度（0-2mm量级），不做Z轴放大
    z_scale_factor = 1.0

    # 原始（未降采样）的点云 + 高度着色（应用 Z 阈值过滤）
    full_points_3d, z_flat = scaled_depth_to_points_and_z(depth, z_scale=z_scale_factor)
    mask_init = z_flat > 0.02  # 示例 main 函数仅用于演示，类内已移除全局阈值
    filtered_points_3d = full_points_3d[mask_init]
    filtered_z_flat = z_flat[mask_init]
    colors = colors_from_height(filtered_z_flat)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Raw Dense Point Cloud (Height-colored)")
    # 增大点大小，便于可视化
    try:
        opt = vis.get_render_option()
        opt.point_size = 3.0
    except Exception:
        pass

    pcd = o3d.geometry.PointCloud()
    # 初次：若非空，设置点云并适配视角；否则先占位添加空几何
    if filtered_points_3d.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(filtered_points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
        try:
            vis.get_view_control().fit_children()
        except Exception:
            pass
    else:
        vis.add_geometry(pcd)

    print("正在显示原始点云（无降采样，高度着色）...")
    print("在弹出的窗口按 'q' 键或关闭窗口来退出程序。")

    keep_running = True
    # 保存上一帧的可见点数，用于从空->非空的视角自适应与重建
    try:
        prev_n = int(np.asarray(pcd.points).shape[0])
    except Exception:
        prev_n = 0
    try:
        while keep_running:
            new_depth = sensor_0.selectSensorInfo(Sensor.OutputType.Depth)
            if new_depth is not None:
                full_points_3d, z_flat = scaled_depth_to_points_and_z(new_depth, z_scale=z_scale_factor)
                # 应用 Z 阈值过滤，仅显示高于阈值的点（示例使用 0.02 mm）
                mask = z_flat > 0.02
                n = int(np.count_nonzero(mask))
                filtered_points_3d = full_points_3d[mask]
                filtered_z_flat = z_flat[mask]
                colors = colors_from_height(filtered_z_flat)

                if n > 0:
                    if prev_n == 0:
                        # 由空变为非空：重建几何并自适应视角
                        try:
                            vis.remove_geometry(pcd, reset_bounding_box=False)
                        except Exception:
                            pass
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filtered_points_3d.astype(np.float64))
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        vis.add_geometry(pcd)
                        try:
                            vis.get_view_control().fit_children()
                        except Exception:
                            pass
                    else:
                        # 正常更新
                        pcd.points = o3d.utility.Vector3dVector(filtered_points_3d.astype(np.float64))
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        vis.update_geometry(pcd)
                else:
                    # 无点可显示：清空
                    empty_pts = np.empty((0, 3), dtype=np.float64)
                    empty_cols = np.empty((0, 3), dtype=np.float64)
                    pcd.points = o3d.utility.Vector3dVector(empty_pts)
                    pcd.colors = o3d.utility.Vector3dVector(empty_cols)
                    vis.update_geometry(pcd)

            if not vis.poll_events():
                keep_running = False
            vis.update_renderer()
            prev_n = int(np.asarray(pcd.points).shape[0])
    finally:
        vis.destroy_window()
        sensor_0.release()
        print("程序已退出。")
        sys.exit()


if __name__ == '__main__':
    main()