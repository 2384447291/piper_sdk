import pyrealsense2 as rs
import numpy as np
import cv2


class RealSense:
    """
    RealSense相机控制类，用于初始化、获取图像和点云数据。
    支持仅深度流、非阻塞取帧与超时配置，从而降低USB带宽占用并避免长时间阻塞。
    """

    # 类级默认外参，允许在未实例化前访问
    _extr_pos_m = np.zeros(3, dtype=np.float64)
    _R_cam_to_world = np.eye(3, dtype=np.float64)

    def __init__(self,
                 width: int = 424,
                 height: int = 240,
                 fps: int = 15,
                 use_color: bool = False,
                 non_blocking: bool = True,
                 timeout_ms: int = 1000):
        """
        初始化RealSense相机

        Args:
            width: 分辨率宽
            height: 分辨率高
            fps: 帧率
            use_color: 是否启用彩色流（默认关闭以降低带宽）
            non_blocking: 是否采用非阻塞取帧（无帧时立即返回空）
            timeout_ms: 阻塞模式下的取帧超时（毫秒）
        """
        print("正在初始化RealSense相机...")
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 配置流：默认仅深度，降低USB带宽
        if use_color:
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # 启动管道（带回退）
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as e:
            print(f"启动RealSense失败，将回退到兼容深度配置: {e}")
            fallback_modes = [
                (640, 480, 30),
                (848, 480, 30),
                (640, 360, 30),
            ]
            started = False
            for fw, fh, ffps in fallback_modes:
                try:
                    cfg_fb = rs.config()
                    cfg_fb.enable_stream(rs.stream.depth, fw, fh, rs.format.z16, ffps)
                    profile = self.pipeline.start(cfg_fb)
                    print(f"已使用兼容模式启动: depth {fw}x{fh}@{ffps}")
                    # 修正内部记录，因实际运行参数已变
                    width, height, fps = fw, fh, ffps
                    self.use_color = False
                    self.align = None
                    started = True
                    break
                except RuntimeError:
                    continue
            if not started:
                raise

        # 获取深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"深度比例系数: {self.depth_scale}")

        # 仅在启用彩色流时做对齐
        self.align = rs.align(rs.stream.color) if use_color else None

        # 记录模式与超时
        self.non_blocking = bool(non_blocking)
        self.timeout_ms = int(timeout_ms)
        self.use_color = bool(use_color)

        # 获取并记录内参（优先彩色，其次深度）
        try:
            if self.non_blocking:
                frames = self.pipeline.poll_for_frames()
            else:
                frames = self.pipeline.wait_for_frames(self.timeout_ms)
            if frames:
                if self.align is not None:
                    frames = self.align.process(frames)
                frame_for_intr = frames.get_color_frame() if self.use_color else frames.get_depth_frame()
                intr = frame_for_intr.profile.as_video_stream_profile().intrinsics
                self.intrinsics = {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "ppx": intr.ppx,
                    "ppy": intr.ppy,
                }
                print(f"相机内参: {self.intrinsics}")
        except Exception as e:
            print(f"获取相机内参失败（不影响运行）: {e}")

        print("RealSense相机初始化完成。")

        # 默认外参（可通过 configure_extrinsics 覆盖）
        if not hasattr(RealSense, "_extr_pos_m"):
            RealSense._extr_pos_m = np.zeros(3, dtype=np.float64)
            RealSense._R_cam_to_world = np.eye(3, dtype=np.float64)

    # ===== 外参/坐标系工具 =====
    @staticmethod
    def _deg2rad(deg: float) -> float:
        return deg * np.pi / 180.0

    @staticmethod
    def _rot_x(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    @staticmethod
    def _rot_y(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    @staticmethod
    def _rot_z(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    @classmethod
    def configure_extrinsics(cls,
                             pos_mm: np.ndarray,
                             yaw_y_deg: float,
                             roll_z_deg: float,
                             pitch_x_deg: float) -> None:
        """配置相机外参（位置毫米，旋转顺序 y(+), z(-), x(-)）并计算 R_cam_to_world。"""
        pos_mm = np.asarray(pos_mm, dtype=np.float64).reshape(3)
        cls._extr_pos_m = pos_mm / 1000.0
        R_world_to_cam = (
            cls._rot_x(cls._deg2rad(pitch_x_deg)) @
            cls._rot_z(cls._deg2rad(roll_z_deg)) @
            cls._rot_y(cls._deg2rad(yaw_y_deg))
        )
        cls._R_cam_to_world = R_world_to_cam.T

    @classmethod
    def get_R_cam_to_world(cls) -> np.ndarray:
        return cls._R_cam_to_world.copy()

    @classmethod
    def get_camera_pos_mm(cls) -> np.ndarray:
        return (cls._extr_pos_m * 1000.0).copy()

    @classmethod
    def transform_points_cam_to_world(cls, points_m: np.ndarray) -> np.ndarray:
        """将相机坐标系下的点(米)转换到世界坐标系(米)。points 形状 (N,3)。"""
        if points_m.size == 0:
            return points_m
        # 标准3D变换公式：world_point = R @ cam_point + t
        # _R_cam_to_world 是 3x3 矩阵，points_m 是 Nx3
        return (cls._R_cam_to_world @ points_m.T).T + cls._extr_pos_m

    @classmethod
    def transform_points_rs_to_world(cls, points_rs_m: np.ndarray) -> np.ndarray:
        """将RealSense原生坐标系下的点(米)直接转换到世界坐标系(米)。
        
        坐标系定义：
        - RealSense原生：Z朝前，X朝右，Y朝下
        - 相机坐标系：Z朝前，X朝右，Y朝上（用户坐标系）
        - 世界坐标系：根据外参定义
        
        转换步骤：
        1. RS原生 -> 相机坐标系（用户坐标系）
        2. 相机坐标系 -> 世界坐标系
        """
        if points_rs_m.size == 0:
            return points_rs_m
        
        # 步骤1：RS原生坐标系 -> 相机坐标系（用户坐标系）
        # RS原生：Z朝前，X朝右，Y朝下
        # 相机坐标系：Z朝前，X朝右，Y朝上
        # 转换：X不变，Y取反，Z不变
        R_rs_to_cam = np.array([
            [1,  0,  0],
            [0, -1,  0], 
            [0,  0,  1]
        ], dtype=np.float64)
        
        points_cam_m = points_rs_m @ R_rs_to_cam.T
        
        # 步骤2：相机坐标系 -> 世界坐标系
        return cls.transform_points_cam_to_world(points_cam_m)

    def get_frames(self):
        """
        获取一对对齐的彩色和深度帧
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image
        except RuntimeError as e:
            print(f"无法获取帧: {e}")
            return None, None

    def get_pointcloud(self, min_distance=0.05, max_distance=0.6):
        """
        获取点云数据，返回RealSense原生坐标系下的点云
        
        Args:
            min_distance: 最小距离(米)，默认5cm
            max_distance: 最大距离(米)，默认60cm
            
        返回: (points, colors) - points为(N,3)的3D坐标（RealSense原生坐标系），colors为(N,3)的RGB颜色
        
        RealSense原生坐标系：Z轴朝前，X轴朝右，Y轴朝下
        """
        try:
            # 获取帧数据（支持非阻塞或带超时的阻塞）
            if self.non_blocking:
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    return np.empty((0, 3)), np.empty((0, 3))
            else:
                frames = self.pipeline.wait_for_frames(self.timeout_ms)

            # 应用滤波器（可选，提高质量）
            # hole_filling = rs.hole_filling_filter()
            # spatial = rs.spatial_filter()
            # temporal = rs.temporal_filter()
            
            if self.align is not None:
                frames = self.align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if self.use_color else None
            
            if not depth_frame or not color_frame:
                # 在仅深度模式下，允许无彩色帧
                if not depth_frame:
                    return np.empty((0, 3)), np.empty((0, 3))
            
            # 创建点云对象
            pc = rs.pointcloud()
            
            # 将颜色映射到点云（仅当存在彩色帧）
            if color_frame is not None:
                pc.map_to(color_frame)
            
            # 计算点云
            points = pc.calculate(depth_frame)
            
            # 获取顶点坐标 (x, y, z)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # 使用RealSense内置的颜色映射（如果可用）
            try:
                # 尝试获取颜色化的点云
                colors = None
                if color_frame is not None:
                    # 若有彩色帧，则按彩色帧生成颜色
                    # 这里简单使用深度归一化配色，若需真彩请改为从 color_frame 采样
                    pass
                
                # 距离过滤：根据参数设置范围
                depth_values = vertices[:, 2]
                valid_mask = (depth_values > min_distance) & (depth_values <= max_distance)
                
                if np.sum(valid_mask) == 0:
                    return np.empty((0, 3)), np.empty((0, 3))
                
                valid_vertices = vertices[valid_mask]
                points_out = valid_vertices  # 直接返回RealSense原生坐标
                valid_depths = depth_values[valid_mask]
                
                # 深度归一化着色（固定方案，避免依赖 colorizer）
                if len(valid_depths) > 0:
                    depth_min, depth_max = valid_depths.min(), valid_depths.max()
                    if depth_max > depth_min:
                        depth_norm = (valid_depths - depth_min) / (depth_max - depth_min)
                    else:
                        depth_norm = np.zeros_like(valid_depths)
                    colors = np.zeros((len(valid_vertices), 3), dtype=np.float32)
                    colors[:, 0] = depth_norm
                    colors[:, 1] = 0.5
                    colors[:, 2] = 1.0 - depth_norm
                else:
                    colors = np.zeros((0, 3), dtype=np.float32)
                
                return points_out, colors
                
            except Exception:
                # 如果颜色化失败，返回基本的深度着色
                valid_mask = (vertices[:, 2] > min_distance) & (vertices[:, 2] <= max_distance)
                points_out = vertices[valid_mask]  # 直接返回RealSense原生坐标
                return points_out, (np.ones((np.sum(valid_mask), 3), dtype=np.float32) * 0.5)
            
        except Exception as e:
            print(f"获取点云失败: {e}")
            return np.empty((0, 3)), np.empty((0, 3))

    def release(self):
        """
        停止管道并释放相机资源
        """
        print("正在停止RealSense相机...")
        self.pipeline.stop()
        print("RealSense相机已停止。")


if __name__ == "__main__":
    # 简单测试
    camera = RealSense()
    try:
        for i in range(5):
            points, colors = camera.get_pointcloud()
            if points is not None:
                print(f"帧 {i+1}: 获得 {points.shape[0]} 个点")
            else:
                print(f"帧 {i+1}: 无点云数据")
    finally:
        camera.release()