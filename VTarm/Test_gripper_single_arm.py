import sys
import time
import termios
import tty
import select
import os
import numpy as np
from typing import Tuple, Optional
from enum import Enum
from tactile_sensor.tactile_sensor import (
    TactileSensorViewer,
    colors_from_height,
    SENSOR_WIDTH_MM,
    SENSOR_HEIGHT_MM,
)

from arm.piper_contol_arm import PiperArmController
from tracking_controller import TrackingController
from visualization.pointcloud_viewer import PointCloudViewer
from realsense.realsense import RealSense


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

OBJECT_PROMPT = "a green bittermelon"
OBJECT_NAME = "bittermelon"

# 相机外参（相对于主坐标系/夹爪中心）：
# 平移（毫米）：后方103.84，右方8.5，上方123.6 → [-103.84, -8.5, 123.6]
CAMERA_POS = [-103.84, 8.5, 123.6]  
YAW_Y_DEG = -28.0  
ROLL_Z_DEG = 90.0
PITCH_X_DEG = 90.0

class _RawTerminal:
    """将终端切换到原始模式，便于无回车读取按键。"""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)

    def __enter__(self) -> "_RawTerminal":
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def _read_key_nonblocking(timeout_s: float = 0.05) -> Tuple[bool, str]:
    """非阻塞读取单个字符，返回(是否读取到, 字符)。"""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if rlist:
        ch = sys.stdin.read(1)
        return True, ch
    return False, ""


class FlowState(Enum):
    CONNECT_ENABLE = 1
    MOVE_TO_ZERO = 2
    START_TRACKING = 3
    WAIT_TRACK_END = 4
    VISUALIZATION = 5
    GRIPPER_CONTROL = 6
    SHUTDOWN = 7


def load_object_model(object_name: str) -> Optional[np.ndarray]:
    """加载物体模型文件"""
    obj_path = f"/home/zenbot-slj/Desktop/piper_arm/Object_data/{object_name}/{object_name}.obj"
    if not os.path.exists(obj_path):
        print(f"[可视化] 物体模型文件不存在: {obj_path}")
        return None
    
    try:
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
        
        if vertices:
            # 转换为numpy数组，并转换为毫米单位
            model_points = np.array(vertices, dtype=np.float64) * 1000.0
            print(f"[可视化] 成功加载物体模型: {object_name}, 顶点数: {len(vertices)}")
            return model_points
        else:
            print(f"[可视化] 物体模型文件为空: {obj_path}")
            return None
    except Exception as e:
        print(f"[可视化] 加载物体模型失败: {e}")
        return None


def transform_model_to_world_frame(model_points: np.ndarray, pose_matrix: np.ndarray) -> np.ndarray:
    """将物体模型变换到世界坐标系
    
    Args:
        model_points: 物体模型点云（毫米，物体自身坐标系）
        pose_matrix: 物体相对于相机坐标系的4x4变换矩阵（米单位）
    
    Returns:
        世界坐标系下的物体点云（毫米）
    """
    # 步骤1：将模型点从毫米转换为米
    model_points_m = model_points / 1000.0
    
    # 步骤2：将模型点从物体坐标系变换到相机坐标系
    homogeneous_points = np.hstack([model_points_m, np.ones((model_points_m.shape[0], 1))])
    camera_frame_points_m = (pose_matrix @ homogeneous_points.T).T
    
    # 步骤3：从相机坐标系变换到世界坐标系
    camera_frame_points_3d = camera_frame_points_m[:, :3]  # 只取前3列（去掉齐次坐标）
    world_frame_points_m = RealSense.transform_points_cam_to_world(camera_frame_points_3d)
    
    # 步骤4：转换回毫米单位
    world_frame_points_mm = world_frame_points_m * 1000.0
    
    return world_frame_points_mm


def setup_visualization() -> Optional[PointCloudViewer]:
    """初始化可视化界面"""
    try:
        viewer = PointCloudViewer(
            window_title="位姿估计结果可视化",
            window_width=1200,
            window_height=800
        )
        
        if not viewer.setup():
            print("[可视化] 可视化界面初始化失败")
            return None
        
        # 添加世界坐标轴
        viewer.add_coordinate_frame("world", size=50)
        
        # 添加相机坐标轴（使用新的外参变换）
        rotation_matrix = RealSense.get_R_cam_to_world()
        viewer.add_coordinate_frame_with_pose(
            name="camera",
            size=20,
            position=RealSense.get_camera_pos_mm().tolist(),
            rotation_matrix=rotation_matrix
        )
        
        print("[可视化] 可视化界面初始化成功")
        return viewer
    except Exception as e:
        print(f"[可视化] 初始化失败: {e}")
        return None


def update_gripper_planes(viewer: PointCloudViewer, gripper_distance_mm: float):
    """更新夹爪对应的两个平面"""
    half_z = gripper_distance_mm / 2.0
    H = float(SENSOR_HEIGHT_MM)
    W = float(SENSOR_WIDTH_MM)
    
    # 左侧传感器边框（Y = -half_z）
    left_corners = np.array([
        [-H/2.0, -half_z, -W/2.0],
        [+H/2.0, -half_z, -W/2.0],
        [+H/2.0, -half_z, +W/2.0],
        [-H/2.0, -half_z, +W/2.0],
    ], dtype=np.float64)
    left_edges = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
    left_colors = np.tile([[0.2,0.8,0.2]], (4,1))  # 绿色
    viewer.add_line_set("left_border", left_corners, left_edges, left_colors)
    
    # 右侧传感器边框（Y = +half_z）
    right_corners = np.array([
        [-H/2.0, +half_z, -W/2.0],
        [+H/2.0, +half_z, -W/2.0],
        [+H/2.0, +half_z, +W/2.0],
        [-H/2.0, +half_z, +W/2.0],
    ], dtype=np.float64)
    right_edges = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
    right_colors = np.tile([[0.8,0.2,0.2]], (4,1))  # 红色
    viewer.add_line_set("right_border", right_corners, right_edges, right_colors)


def extract_gripper_contact_surfaces(object_points_mm: np.ndarray, gripper_distance_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """提取夹爪接触面点云
    
    Args:
        object_points_mm: 物体点云（毫米，世界坐标系）
        gripper_distance_mm: 夹爪间距
    
    Returns:
        (left_contact_points, right_contact_points): 左右接触面点云
    """
    if object_points_mm.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    half_z = gripper_distance_mm / 2.0
    H_expanded = SENSOR_HEIGHT_MM * 1.5  # 扩大1.5倍
    W_expanded = SENSOR_WIDTH_MM * 1.5   # 扩大1.5倍
    
    # 定义扩大后的夹爪平面范围
    x_range = [-H_expanded/2.0, +H_expanded/2.0]
    z_range = [-W_expanded/2.0, +W_expanded/2.0]
    
    # 左侧夹爪接触面（Y = -half_z 向内）
    left_contact_points = []
    # 右侧夹爪接触面（Y = +half_z 向内）
    right_contact_points = []
    
    # 在扩大的平面范围内采样网格点
    x_samples = np.linspace(x_range[0], x_range[1], 20)
    z_samples = np.linspace(z_range[0], z_range[1], 20)
    
    for x in x_samples:
        for z in z_samples:
            # 左侧：从 Y = -half_z 向内（正Y方向）投射
            left_ray_start = np.array([x, -half_z, z])
            left_ray_dir = np.array([0, 1, 0])  # 向内（正Y方向）
            
            # 右侧：从 Y = +half_z 向内（负Y方向）投射
            right_ray_start = np.array([x, +half_z, z])
            right_ray_dir = np.array([0, -1, 0])  # 向内（负Y方向）
            
            # 找到射线与物体的第一个交点
            left_contact = find_first_intersection(object_points_mm, left_ray_start, left_ray_dir)
            right_contact = find_first_intersection(object_points_mm, right_ray_start, right_ray_dir)
            
            if left_contact is not None:
                left_contact_points.append(left_contact)
            if right_contact is not None:
                right_contact_points.append(right_contact)
    
    left_contact_array = np.array(left_contact_points) if left_contact_points else np.empty((0, 3))
    right_contact_array = np.array(right_contact_points) if right_contact_points else np.empty((0, 3))
    
    return left_contact_array, right_contact_array


def find_first_intersection(points: np.ndarray, ray_start: np.ndarray, ray_dir: np.ndarray, 
                          search_distance: float = 200.0, tolerance: float = 2.0) -> Optional[np.ndarray]:
    """沿射线方向找到第一个与点云的交点
    
    Args:
        points: 点云数组 (N, 3)
        ray_start: 射线起点
        ray_dir: 射线方向（单位向量）
        search_distance: 搜索距离（毫米）
        tolerance: 交点容差（毫米）
    
    Returns:
        第一个交点坐标，如果没有找到则返回None
    """
    if points.shape[0] == 0:
        return None
    
    # 沿射线方向搜索
    min_distance = float('inf')
    closest_point = None
    
    # 计算所有点到射线的距离和沿射线方向的投影
    for i in range(0, len(points), max(1, len(points) // 1000)):  # 采样以提高效率
        point = points[i]
        to_point = point - ray_start
        
        # 计算沿射线方向的投影距离
        proj_distance = np.dot(to_point, ray_dir)
        
        # 只考虑射线前方的点
        if proj_distance > 0 and proj_distance < search_distance:
            # 计算点到射线的垂直距离
            proj_point = ray_start + proj_distance * ray_dir
            perp_distance = np.linalg.norm(point - proj_point)
            
            # 如果点在容差范围内且更近
            if perp_distance < tolerance and proj_distance < min_distance:
                min_distance = proj_distance
                closest_point = point
    
    return closest_point


def main() -> None:
    controller = PiperArmController()
    tracker = TrackingController(port=10000, authkey=b"foundationpose")
    # 配置 RealSense 外参，供全局使用
    RealSense.configure_extrinsics(
        pos_mm=CAMERA_POS,
        yaw_y_deg=YAW_Y_DEG,
        roll_z_deg=ROLL_Z_DEG,
        pitch_x_deg=PITCH_X_DEG,
    )

    target_joints_deg = [0.0, 100.0, -30.0, 0.0, -70.0, 0.0]
    gripper_mm = 60.0
    fine_step_mm = 0.05
    coarse_step_mm = 0.2
    min_mm = -10.0
    max_mm = 80.0

    object_prompt = OBJECT_PROMPT
    object_name = OBJECT_NAME

    state = FlowState.CONNECT_ENABLE
    viewer = None
    avg_pose = None
    model_points = None
    left_contact_points = None
    right_contact_points = None
    left_contact_colors = None
    right_contact_colors = None
    transformed_model = None
    model_colors = None

    print("[操作] 按键：k 结束跟踪，g 切换到纯夹爪控制，q/w 细调(0.05mm)，a/d 粗调(0.2mm)，ESC 退出")
    print("[提示] 可视化状态下夹爪控制仍然有效，平面间距会实时更新")

    with _RawTerminal():
        try:
            while True:
                # 非阻塞按键读取
                has_key, key = _read_key_nonblocking(timeout_s=0.01)

                if has_key and key == "\x1b":  # ESC 退出
                    state = FlowState.SHUTDOWN

                # 状态机
                if state == FlowState.CONNECT_ENABLE:
                    print("[提示] 正在连接机械臂...")
                    controller.connect()
                    if not controller.enable(timeout_s=5.0):
                        print("[错误] 使能失败：请检查电源/CAN 连接")
                        state = FlowState.SHUTDOWN
                        continue
                    print("[成功] 机械臂已连接并使能")
                    controller.set_joint_mode(speed_percent=30, is_mit_mode=0x00)
                    state = FlowState.MOVE_TO_ZERO

                elif state == FlowState.MOVE_TO_ZERO:
                    print("[提示] 正在回到目标位姿...")
                    controller.move_to_zero(
                        target_joint_angles=target_joints_deg,
                        target_gripper_value=gripper_mm,
                        joint_in_radians=False,
                        gripper_is_normalized=False,
                        speed_percent=30,
                        iterations=20,
                        iteration_interval_s=0.05,
                    )
                    reached = controller.wait_until_reached(
                        target_joint_angles=target_joints_deg,
                        target_gripper_value=gripper_mm,
                        joint_in_radians=False,
                        gripper_is_normalized=False,
                        tolerance=2,
                        timeout_s=10.0,
                        check_interval_s=0.2,
                    )
                    print("[结果] 回零:", "成功" if reached else "超时")
                    state = FlowState.START_TRACKING

                elif state == FlowState.START_TRACKING:
                    try:
                        tracker.start()
                        tracker.send_start(object_prompt=object_prompt, object_name=object_name)
                        print("[Tracking] 已发送开始跟踪请求，按 'k' 结束并计算平均位姿")
                        state = FlowState.WAIT_TRACK_END
                    except Exception as e:
                        print(f"[Tracking][错误] 无法启动或发送开始命令: {e}")
                        state = FlowState.SHUTDOWN

                elif state == FlowState.WAIT_TRACK_END:
                    # 仅监听按键 'k' 来结束跟踪
                    if has_key and key.lower() == "k":
                        try:
                            print("[Tracking] 停止跟踪并计算平均位姿...")
                            avg_pose = tracker.request_average_pose(expected_samples=20, wait_timeout=10.0)
                            if avg_pose is not None:
                                t = avg_pose[:3, 3]
                                print(f"[Tracking] 平均位姿(相机坐标系): 平移=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) 米")
                                print(f"[Tracking] 平均位姿: 旋转=({avg_pose[:3, :3]})") 
                                state = FlowState.VISUALIZATION
                            else:
                                print("[Tracking] 未获得平均位姿，直接进入夹爪控制")
                                state = FlowState.GRIPPER_CONTROL
                        except Exception as e:
                            print(f"[Tracking][错误] 计算平均位姿失败: {e}")
                            state = FlowState.GRIPPER_CONTROL

                elif state == FlowState.VISUALIZATION:
                    # 初始化可视化界面
                    if viewer is None:
                        viewer = setup_visualization()
                        if viewer is None:
                            print("[可视化] 可视化初始化失败，直接进入夹爪控制")
                            state = FlowState.GRIPPER_CONTROL
                            continue
                        
                        # 加载物体模型
                        model_points = load_object_model(object_name)
                        if model_points is None:
                            print("[可视化] 物体模型加载失败，直接进入夹爪控制")
                            state = FlowState.GRIPPER_CONTROL
                            continue
                        
                        # 一次性计算接触面（物体静止，只需计算一次）
                        transformed_model = transform_model_to_world_frame(model_points, avg_pose)
                        model_colors = np.tile([[0.0, 0.0, 1.0]], (transformed_model.shape[0], 1))  # 蓝色
                        
                        print("[可视化] 正在计算夹爪接触面...")
                        left_contact_points, right_contact_points = extract_gripper_contact_surfaces(transformed_model, gripper_mm)
                        
                        # 预计算颜色
                        if left_contact_points.shape[0] > 0:
                            left_contact_colors = np.tile([[1.0, 0.5, 0.0]], (left_contact_points.shape[0], 1))  # 橙色
                        if right_contact_points.shape[0] > 0:
                            right_contact_colors = np.tile([[0.8, 0.0, 0.8]], (right_contact_points.shape[0], 1))  # 紫色
                        
                        print(f"[可视化] 接触面计算完成：左侧{left_contact_points.shape[0]}点，右侧{right_contact_points.shape[0]}点")
                    
                    # 处理夹爪控制按键（在可视化状态下仍然生效）
                    if has_key:
                        if key.lower() == "w":
                            gripper_mm = _clamp(gripper_mm + fine_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "q":
                            gripper_mm = _clamp(gripper_mm - fine_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "d":
                            gripper_mm = _clamp(gripper_mm + coarse_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "a":
                            gripper_mm = _clamp(gripper_mm - coarse_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "g":
                            print("[可视化] 切换到纯夹爪控制模式")
                            state = FlowState.GRIPPER_CONTROL
                    
                    # 更新可视化
                    if viewer.is_running():
                        # 更新夹爪平面（根据当前夹爪距离）
                        update_gripper_planes(viewer, gripper_mm)
                        
                        # 显示预计算的物体模型和接触面（只需要渲染，不需要重新计算）
                        if transformed_model is not None and model_colors is not None:
                            viewer.update_pointcloud("object_model", transformed_model, model_colors)
                            
                            # 可视化预计算的接触面
                            if left_contact_points is not None and left_contact_colors is not None and left_contact_points.shape[0] > 0:
                                viewer.update_pointcloud("left_contact", left_contact_points, left_contact_colors)
                            
                            if right_contact_points is not None and right_contact_colors is not None and right_contact_points.shape[0] > 0:
                                viewer.update_pointcloud("right_contact", right_contact_points, right_contact_colors)
                        
                        # 运行更新循环
                        if not viewer.run_update_cycle():
                            print("[可视化] 可视化窗口已关闭，进入夹爪控制")
                            state = FlowState.GRIPPER_CONTROL
                    else:
                        print("[可视化] 可视化窗口已关闭，进入夹爪控制")
                        state = FlowState.GRIPPER_CONTROL

                elif state == FlowState.GRIPPER_CONTROL:
                    if has_key:
                        if key.lower() == "w":
                            gripper_mm = _clamp(gripper_mm + fine_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "q":
                            gripper_mm = _clamp(gripper_mm - fine_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "d":
                            gripper_mm = _clamp(gripper_mm + coarse_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)
                        elif key.lower() == "a":
                            gripper_mm = _clamp(gripper_mm - coarse_step_mm, min_mm, max_mm)
                            controller.command_gripper(gripper_value=gripper_mm, gripper_is_normalized=False)

                elif state == FlowState.SHUTDOWN:
                    print("[提示] 正在退出...")
                    break

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("[提示] 捕获到 Ctrl+C，正在退出...")
        finally:
            try:
                tracker.stop()
            except Exception:
                pass
            
            # 清理可视化资源
            if viewer is not None:
                try:
                    viewer.close()
                    print("[可视化] 可视化界面已关闭")
                except Exception as e:
                    print(f"[可视化] 关闭可视化界面时出错: {e}")


if __name__ == "__main__":
    main()


