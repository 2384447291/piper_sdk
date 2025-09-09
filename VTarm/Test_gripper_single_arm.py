import sys
import time
import termios
import tty
import select
import os
import imageio.v2 as imageio
import numpy as np
import threading
import queue
from typing import Tuple, Optional
from enum import Enum
from tactile_sensor.tactile_sensor import (
    TactileSensor,
    SENSOR_WIDTH_MM,
    SENSOR_HEIGHT_MM,
)

from arm.piper_contol_arm import PiperArmController
from tracking_controller import TrackingController
from visualization import Viewer
from realsense.realsense import RealSense
from shared_data import SharedData, FlowState


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

OBJECT_PROMPT = "a green bittermelon"
OBJECT_NAME = "bittermelon"
SAMPLES_ROOT = "/home/zenbot-slj/Desktop/piper_arm/tactile_samples"  # 采样主目录

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


def _flush_input_buffer():
    """清空输入缓冲区，防止按键积压"""
    while True:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)  # 立即返回
        if not rlist:
            break
        sys.stdin.read(1)  # 丢弃缓冲区中的字符




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


def setup_visualization() -> Optional[Viewer]:
    """初始化可视化界面"""
    try:
        viewer = Viewer(
            window_title="位姿估计结果可视化",
            window_width=1200,
            window_height=800,
            num_2d_panels=2,
            o3d_ratio=0.65,
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


def update_gripper_planes(viewer: Viewer, gripper_distance_mm: float):
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


def extract_gripper_contact_surfaces(object_points_mm: np.ndarray,
                                     plane_tolerance_mm: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取夹爪接触轮廓并为所有物体点云添加接触标记

    新逻辑：忽略与夹爪平面的 y 距离阈值，仅要求点的 x/z 落在夹爪平面投影范围内；
    使用物体点云在 y 轴的中位数作为"物体中心"，将 y 小于中心的点视为左轮廓，
    y 大于等于中心的点视为右轮廓，避免平面穿模带来的误判。
    
    为所有物体点云添加第四维接触标记：
    - 0: 未被接触
    - 1: 左夹爪接触
    - 2: 右夹爪接触

    Args:
        object_points_mm: 物体点云（毫米，世界坐标系）(N, 3)
        plane_tolerance_mm: 接触面厚度容忍度

    Returns:
        (left_contact_points, right_contact_points, object_points_with_contact): 
        - 左接触面点云 (M, 3)
        - 右接触面点云 (K, 3) 
        - 带接触标记的物体点云 (N, 4) [x, y, z, contact_label]
    """
    if object_points_mm.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 4))

    # 扩大后的平面范围（与原视觉边界一致）
    H_expanded = SENSOR_HEIGHT_MM * 1.2
    W_expanded = SENSOR_WIDTH_MM * 1.5

    x_min, x_max = -H_expanded / 2.0, +H_expanded / 2.0
    z_min, z_max = -W_expanded / 2.0, +W_expanded / 2.0

    x = object_points_mm[:, 0]
    y = object_points_mm[:, 1]
    z = object_points_mm[:, 2]

    # 初始化接触标记（0: 未接触）
    contact_labels = np.zeros(object_points_mm.shape[0], dtype=np.int32)

    # 先筛选落在夹爪平面投影范围内的所有点（xz 内）
    inside_mask = (
        (x >= x_min) & (x <= x_max) &
        (z >= z_min) & (z <= z_max)
    )

    if not np.any(inside_mask):
        # 如果没有点在夹爪范围内，返回带标记的原始点云
        object_points_with_contact = np.column_stack((object_points_mm, contact_labels))
        return np.empty((0, 3)), np.empty((0, 3)), object_points_with_contact

    # 用 y 的中位数作为物体中心（在投影内点上计算），鲁棒于离群点
    center_y = np.median(y[inside_mask])

    # 先按中心划分左右候选
    left_candidates = inside_mask & (y < center_y)
    right_candidates = inside_mask & (y >= center_y)

    left_contact_points = np.empty((0, 3))
    right_contact_points = np.empty((0, 3))

    # 左轮廓：取左侧候选中最左边（y 最小）附近 tolerance 范围内的点
    if np.any(left_candidates):
        y_left = y[left_candidates]
        y_left_min = np.min(y_left)  # 最左边的点
        left_edge_mask = left_candidates & (y <= (y_left_min + plane_tolerance_mm))
        
        # 标记左夹爪接触点（标记为1）
        contact_labels[left_edge_mask] = 1
        left_contact_points = object_points_mm[left_edge_mask]

    # 右轮廓：取右侧候选中最右边（y 最大）附近 tolerance 范围内的点
    if np.any(right_candidates):
        y_right = y[right_candidates]
        y_right_max = np.max(y_right)  # 最右边的点
        right_edge_mask = right_candidates & (y >= (y_right_max - plane_tolerance_mm))
        
        # 标记右夹爪接触点（标记为2）
        contact_labels[right_edge_mask] = 2
        right_contact_points = object_points_mm[right_edge_mask]

    # 创建带接触标记的物体点云 (N, 4)
    object_points_with_contact = np.column_stack((object_points_mm, contact_labels))

    return left_contact_points, right_contact_points, object_points_with_contact






def keyboard_control_thread(shared_data: SharedData, controller: PiperArmController, tracker: TrackingController):
    """键盘控制线程 - 200Hz"""
    fine_step_mm = 0.1
    coarse_step_mm = 0.4
    min_mm = 0.0
    max_mm = 70.0
    
    def _next_sample_index(root_dir: str, object_name: str) -> int:
        try:
            if not os.path.isdir(root_dir):
                return 0
            prefix = f"{object_name}_"
            indices = []
            for name in os.listdir(root_dir):
                if name.startswith(prefix):
                    try:
                        idx = int(name[len(prefix):])
                        indices.append(idx)
                    except Exception: 
                        pass
            return (max(indices) + 1) if indices else 0
        except Exception:
            return 0
    
    with _RawTerminal():
        while not shared_data.shutdown:
            has_key, key = _read_key_nonblocking(timeout_s=0.005)  # 200Hz
            
            if has_key and key == "\x1b":  # ESC 退出
                with shared_data.lock:
                    shared_data.shutdown = True
                break
                
            with shared_data.lock:
                if shared_data.state == FlowState.WAIT_TRACK_END and has_key:
                    if key.lower() == "k":
                        # 结束跟踪并计算平均位姿
                        try:
                            print("[Tracking] 停止跟踪并计算平均位姿...")
                            avg_pose = tracker.request_average_pose(expected_samples=10, wait_timeout=20.0)
                            if avg_pose is not None:
                                t = avg_pose[:3, 3]
                                print(f"[Tracking] 平均位姿(相机坐标系): 平移=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) 米")
                                print(f"[Tracking] 平均位姿: 旋转=({avg_pose[:3, :3]})") 
                                shared_data.avg_pose = avg_pose
                                shared_data.state = FlowState.VISUALIZATION
                            else:
                                print("[Tracking] 未获得平均位姿，直接进入夹爪控制")
                                shared_data.state = FlowState.GRIPPER_CONTROL
                        except Exception as e:
                            print(f"[Tracking][错误] 计算平均位姿失败: {e}")
                            shared_data.state = FlowState.GRIPPER_CONTROL
                    elif key.lower() == "q":
                        # 全局重启（最高优先级）
                        print("[Tracking] 请求全局重启...")
                        shared_data.restart_requested = True
                
                elif shared_data.state == FlowState.VISUALIZATION and has_key:
                    if key == " ":  # 空格键
                        print("[可视化] 满意当前结果，进入夹爪控制阶段")
                        # 延迟关闭3D可视化窗口，避免段错误
                        try:
                            if shared_data.viewer is not None:
                                print("[可视化] 正在关闭3D窗口...")
                                # 先停止更新循环
                                shared_data.viewer.running = False
                                time.sleep(0.1)  # 给一点时间让更新循环结束
                                shared_data.viewer.close()
                                shared_data.viewer = None
                                print("[可视化] 3D窗口已关闭")
                        except Exception as e:
                            print(f"[可视化] 关闭3D窗口失败: {e}")
                        shared_data.state = FlowState.GRIPPER_CONTROL
                    elif key.lower() == "q":
                        # 全局重启（最高优先级）
                        print("[可视化] 请求全局重启...")
                        # 仅尽力关闭UI，避免长时间持锁
                        try:
                            if shared_data.viewer is not None:
                                print("[可视化] 正在关闭3D窗口...")
                                shared_data.viewer.running = False
                                time.sleep(0.05)
                                shared_data.viewer.close()
                                shared_data.viewer = None
                                print("[可视化] 3D窗口已关闭")
                        except Exception:
                            pass
                        shared_data.restart_requested = True
                
                elif shared_data.state == FlowState.GRIPPER_CONTROL and has_key:
                    if key.lower() == "w":
                        shared_data.gripper_mm = _clamp(shared_data.gripper_mm + fine_step_mm, min_mm, max_mm)
                        controller.command_gripper(gripper_value=shared_data.gripper_mm)
                        print(f"[夹爪控制] 细调增加 -> {shared_data.gripper_mm:.2f}mm")
                        _flush_input_buffer()  # 清空缓冲区，防止按键积压
                    elif key.lower() == "s":
                        shared_data.gripper_mm = _clamp(shared_data.gripper_mm - fine_step_mm, min_mm, max_mm)
                        controller.command_gripper(gripper_value=shared_data.gripper_mm)
                        print(f"[夹爪控制] 细调减少 -> {shared_data.gripper_mm:.2f}mm")
                        _flush_input_buffer()  # 清空缓冲区，防止按键积压
                    elif key.lower() == "d":
                        shared_data.gripper_mm = _clamp(shared_data.gripper_mm + coarse_step_mm, min_mm, max_mm)
                        controller.command_gripper(gripper_value=shared_data.gripper_mm)
                        print(f"[夹爪控制] 粗调增加 -> {shared_data.gripper_mm:.2f}mm")
                        _flush_input_buffer()  # 清空缓冲区，防止按键积压
                    elif key.lower() == "a":
                        shared_data.gripper_mm = _clamp(shared_data.gripper_mm - coarse_step_mm, min_mm, max_mm)
                        controller.command_gripper(gripper_value=shared_data.gripper_mm)
                        print(f"[夹爪控制] 粗调减少 -> {shared_data.gripper_mm:.2f}mm")
                        _flush_input_buffer()  # 清空缓冲区，防止按键积压
                    elif key == " ":
                        # 切换采样
                        if not shared_data.is_sampling:
                            # 开始采样
                            try:
                                os.makedirs(SAMPLES_ROOT, exist_ok=True)
                                idx = _next_sample_index(SAMPLES_ROOT, OBJECT_NAME)
                                shared_data.sample_dir = os.path.join(SAMPLES_ROOT, f"{OBJECT_NAME}_{idx}")
                                shared_data.left_rectify_dir = os.path.join(shared_data.sample_dir, 'left_rectify')
                                shared_data.left_difference_dir = os.path.join(shared_data.sample_dir, 'left_difference')
                                shared_data.right_rectify_dir = os.path.join(shared_data.sample_dir, 'right_rectify')
                                shared_data.right_difference_dir = os.path.join(shared_data.sample_dir, 'right_difference')
                                os.makedirs(shared_data.left_rectify_dir, exist_ok=True)
                                os.makedirs(shared_data.left_difference_dir, exist_ok=True)
                                os.makedirs(shared_data.right_rectify_dir, exist_ok=True)
                                os.makedirs(shared_data.right_difference_dir, exist_ok=True)
                                # 保存位姿阶段得到的4维点云（若可用）
                                try:
                                    if shared_data.object_points_with_contact is not None:
                                        np.save(os.path.join(shared_data.sample_dir, 'object_points_with_contact.npy'), shared_data.object_points_with_contact)
                                except Exception as e:
                                    print(f"[采样][警告] 保存4维点云失败: {e}")
                                # 初始化日志
                                with open(os.path.join(shared_data.sample_dir, 'gripper_log.csv'), 'w') as f:
                                    f.write('timestamp_ms,cmd_mm,fb_mm\n')
                                with open(os.path.join(shared_data.sample_dir, 'meta.txt'), 'w') as f:
                                    f.write(f'object_name={OBJECT_NAME}\n')
                                shared_data.sample_start_time = time.time()
                                shared_data.is_sampling = True
                                print(f"[采样] 开始 -> {shared_data.sample_dir}")
                            except Exception as e:
                                print(f"[采样][错误] 启动失败: {e}")
                        else:
                            # 停止采样
                            shared_data.is_sampling = False
                            print("[采样] 已停止")
                    elif key.lower() == "q":
                        # 全局重启（最高优先级）
                        if shared_data.is_sampling:
                            shared_data.is_sampling = False
                            print("[采样] 已停止")
                        print("[夹爪控制] 请求全局重启...")
                        # 请求主线程关闭2D窗口
                        shared_data.viewer_close_requested = True
                        shared_data.restart_requested = True


def tactile_data_thread(shared_data: SharedData):
    """触觉数据获取线程 - 30Hz"""
    last_tactile_ts = 0.0
    tactile_frame_interval = 1.0 / 30.0
    
    while not shared_data.shutdown:
        with shared_data.lock:
            if shared_data.state == FlowState.GRIPPER_CONTROL:
                # 以30Hz获取触觉数据
                now_ts = time.time()
                if now_ts - last_tactile_ts >= tactile_frame_interval:
                    if shared_data.tactile_left is not None and shared_data.tactile_right is not None:
                        try:
                            # 获取四张图
                            left_rectify = shared_data.tactile_left.rectify()
                            right_rectify = shared_data.tactile_right.rectify()
                            left_difference = shared_data.tactile_left.difference()
                            right_difference = shared_data.tactile_right.difference()
                            
                            # 更新共享数据（rectify和difference本身就是uint8格式）
                            shared_data.left_rectify = left_rectify
                            shared_data.right_rectify = right_rectify
                            shared_data.left_difference = left_difference
                            shared_data.right_difference = right_difference
                            
                        except Exception as e:
                            print(f"[触觉] 数据获取失败: {e}")
                    last_tactile_ts = now_ts
        
        time.sleep(1.0 / 30.0)  # 30Hz


def data_acquisition_thread(shared_data: SharedData, controller: PiperArmController):
    """数据获取线程 - 60Hz"""
    last_sample_ts = 0.0
    sample_interval = 1.0 / 60.0
    
    while not shared_data.shutdown:
        with shared_data.lock:
            if shared_data.state == FlowState.GRIPPER_CONTROL and shared_data.is_sampling:
                now_ts = time.time()
                if now_ts - last_sample_ts >= sample_interval:
                    try:
                        t_ms = int((time.time() - shared_data.sample_start_time) * 1000)
                        
                        # 保存四张图像到四个独立文件夹（直接使用原始uint8数据）
                        if shared_data.left_rectify is not None and shared_data.left_rectify_dir is not None:
                            imageio.imwrite(os.path.join(shared_data.left_rectify_dir, f"{t_ms}.png"), shared_data.left_rectify)
                        if shared_data.left_difference is not None and shared_data.left_difference_dir is not None:
                            imageio.imwrite(os.path.join(shared_data.left_difference_dir, f"{t_ms}.png"), shared_data.left_difference)
                        if shared_data.right_rectify is not None and shared_data.right_rectify_dir is not None:
                            imageio.imwrite(os.path.join(shared_data.right_rectify_dir, f"{t_ms}.png"), shared_data.right_rectify)
                        if shared_data.right_difference is not None and shared_data.right_difference_dir is not None:
                            imageio.imwrite(os.path.join(shared_data.right_difference_dir, f"{t_ms}.png"), shared_data.right_difference)
                        
                        # 记录夹爪命令与反馈
                        cmd_val = shared_data.gripper_mm
                        fb_val = None
                        try:
                            # 使用get_joint_angles获取夹爪反馈（第7个元素）
                            joint_angles = controller.get_joint_angles(in_radians=True)
                            if len(joint_angles) >= 7:
                                fb_val = joint_angles[6]  # 夹爪反馈在第7个位置（索引6）
                        except Exception as e:
                            print(f"[采样][调试] 获取夹爪反馈失败: {e}")
                            fb_val = None
                        
                        with open(os.path.join(shared_data.sample_dir, 'gripper_log.csv'), 'a') as f:
                            f.write(f"{t_ms},{cmd_val},{fb_val if fb_val is not None else ''}\n")
                    except Exception as e:
                        print(f"[采样][错误] 保存失败: {e}")
                    last_sample_ts = now_ts
        
        time.sleep(1.0 / 60.0)  # 60Hz


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

    target_joints_deg = [0.0, 100.0, -30.0, 0.0, -60.0, 0.0]
    fine_step_mm = 0.05
    coarse_step_mm = 0.4
    min_mm = 0.0
    max_mm = 70.0

    object_prompt = OBJECT_PROMPT
    object_name = OBJECT_NAME

    # 创建共享数据
    shared_data = SharedData()
    shared_data.gripper_mm = 70.0

    def _next_sample_index(root_dir: str, object_name: str) -> int:
        try:
            if not os.path.isdir(root_dir):
                return 0
            prefix = f"{object_name}_"
            indices = []
            for name in os.listdir(root_dir):
                if name.startswith(prefix):
                    try:
                        idx = int(name[len(prefix):])
                        indices.append(idx)
                    except Exception: 
                        pass
            return (max(indices) + 1) if indices else 0
        except Exception:
            return 0

    print("[操作] 按键说明：")
    print("  跟踪阶段 - k: 结束跟踪计算平均位姿, q: 回到跟踪阶段")
    print("  可视化阶段 - 空格: 满意当前结果进入夹爪控制, q: 回到跟踪阶段")
    print("  夹爪控制 - w/s: 细调(0.05mm), a/d: 粗调(0.4mm), 空格: 采样开始/停止, q: 回到跟踪阶段")
    print("  全局 - ESC: 退出程序")

    # 启动所有子线程（程序开始就启动）
    keyboard_thread = threading.Thread(target=keyboard_control_thread, args=(shared_data, controller, tracker), daemon=True)
    tactile_thread = threading.Thread(target=tactile_data_thread, args=(shared_data,), daemon=True)
    data_thread = threading.Thread(target=data_acquisition_thread, args=(shared_data, controller), daemon=True)
    
    # 立即启动所有线程
    keyboard_thread.start()
    tactile_thread.start()
    data_thread.start()
    print("[系统] 所有子线程已启动")
    
    try:
        # 主线程处理状态机
        while not shared_data.shutdown:
            # 全局重启优先级最高：主线程串行清理并重置
            if getattr(shared_data, 'restart_requested', False) and not getattr(shared_data, 'restarting', False):
                print("[重启] 收到全局重启请求，开始重启流程...")
                shared_data.restarting = True
                try:
                    # 停止tracker（容错）
                    try:
                        tracker.stop()
                    except Exception:
                        pass
                    # 清理所有资源
                    shared_data.cleanup_all()
                    # 重置状态
                    with shared_data.lock:
                        shared_data.state = FlowState.CONNECT_ENABLE
                        shared_data.restart_requested = False
                    print("[重启] 重启流程完成，回到CONNECT_ENABLE")
                finally:
                    shared_data.restarting = False
                # 继续下一轮循环
                time.sleep(0.05)
                continue

            with shared_data.lock:
                current_state = shared_data.state
            
            if current_state == FlowState.CONNECT_ENABLE:
                print("[提示] 正在连接机械臂...")
                controller.connect()
                if not controller.enable(timeout_s=5.0):
                    print("[错误] 使能失败：请检查电源/CAN 连接")
                    shared_data.shutdown = True
                    break
                print("[成功] 机械臂已连接并使能")
                controller.set_joint_mode(speed_percent=30, is_mit_mode=0x00)
                with shared_data.lock:
                    shared_data.state = FlowState.MOVE_TO_ZERO

            elif current_state == FlowState.MOVE_TO_ZERO:
                # 确保从其他状态切换到MOVE_TO_ZERO时清理资源
                with shared_data.lock:
                    if shared_data.viewer is not None:
                        print("[MOVE_TO_ZERO] 检测到viewer仍存在，正在清理...")
                        try:
                            shared_data.viewer.running = False
                            shared_data.viewer.close()
                            shared_data.viewer = None
                            print("[MOVE_TO_ZERO] viewer已清理")
                        except Exception as e:
                            print(f"[MOVE_TO_ZERO] 清理viewer失败: {e}")
                            shared_data.viewer = None
                
                print("[提示] 正在回到目标位姿...")
                controller.move_to_zero(
                    target_joint_angles=target_joints_deg,
                    target_gripper_value=shared_data.gripper_mm,
                    joint_in_radians=False,
                    speed_percent=30,
                    iterations=20,
                    iteration_interval_s=0.05,
                )
                reached = controller.wait_until_reached(
                    target_joint_angles=target_joints_deg,
                    target_gripper_value=shared_data.gripper_mm,
                    joint_in_radians=False,
                    tolerance=2,
                    timeout_s=10.0,
                    check_interval_s=0.2,
                )
                print("[结果] 回零:", "成功" if reached else "超时")
                with shared_data.lock:
                    shared_data.state = FlowState.START_TRACKING

            elif current_state == FlowState.START_TRACKING:
                print("[状态机] 进入START_TRACKING状态")
                try:
                    print("[Tracking] 正在启动tracker...")
                    tracker.start()
                    print("[Tracking] 正在发送开始跟踪请求...")
                    tracker.send_start(object_prompt=object_prompt, object_name=object_name)
                    print("[Tracking] 已发送开始跟踪请求，按 'k' 结束并计算平均位姿")
                    with shared_data.lock:
                        shared_data.state = FlowState.WAIT_TRACK_END
                    print("[状态机] 状态已切换到WAIT_TRACK_END")
                except Exception as e:
                    print(f"[Tracking][错误] 无法启动或发送开始命令: {e}")
                    shared_data.shutdown = True
                    break

            elif current_state == FlowState.WAIT_TRACK_END:
                # 等待键盘线程处理按键
                time.sleep(0.1)

            elif current_state == FlowState.VISUALIZATION:
                # 初始化可视化界面（仅一次）
                with shared_data.lock:
                    if shared_data.viewer is None:
                        shared_data.viewer = setup_visualization()
                        if shared_data.viewer is None:
                            print("[可视化] 可视化初始化失败，直接进入夹爪控制")
                            shared_data.state = FlowState.GRIPPER_CONTROL
                            continue
                        
                        # 加载物体模型
                        shared_data.model_points = load_object_model(object_name)
                        if shared_data.model_points is None:
                            print("[可视化] 物体模型加载失败，直接进入夹爪控制")
                            shared_data.state = FlowState.GRIPPER_CONTROL
                            continue
                        
                        # 一次性计算接触面（物体静止，只需计算一次）
                        shared_data.transformed_model = transform_model_to_world_frame(shared_data.model_points, shared_data.avg_pose)
                        shared_data.model_colors = np.tile([[0.0, 0.0, 1.0]], (shared_data.transformed_model.shape[0], 1))  # 蓝色
                        
                        print("[可视化] 正在计算夹爪接触面...")
                        shared_data.left_contact_points, shared_data.right_contact_points, shared_data.object_points_with_contact = extract_gripper_contact_surfaces(shared_data.transformed_model, 10)
                        
                        # 预计算颜色：左侧绿色，右侧红色   
                        if shared_data.left_contact_points.shape[0] > 0:
                            shared_data.left_contact_colors = np.tile([[0.0, 1.0, 0.0]], (shared_data.left_contact_points.shape[0], 1))  # 绿色
                        if shared_data.right_contact_points.shape[0] > 0:
                            shared_data.right_contact_colors = np.tile([[1.0, 0.0, 0.0]], (shared_data.right_contact_points.shape[0], 1))  # 红色
                        
                        print(f"[可视化] 接触面计算完成：左侧{shared_data.left_contact_points.shape[0]}点，右侧{shared_data.right_contact_points.shape[0]}点")
                        # 调试：打印空间范围，确认是否在视野内
                        if shared_data.left_contact_points.shape[0] > 0:
                            lmin = shared_data.left_contact_points.min(axis=0)
                            lmax = shared_data.left_contact_points.max(axis=0)
                            print(f"[调试] 左接触范围 x[{lmin[0]:.1f},{lmax[0]:.1f}] y[{lmin[1]:.1f},{lmax[1]:.1f}] z[{lmin[2]:.1f},{lmax[2]:.1f}]")
                        if shared_data.right_contact_points.shape[0] > 0:
                            rmin = shared_data.right_contact_points.min(axis=0)
                            rmax = shared_data.right_contact_points.max(axis=0)
                            print(f"[调试] 右接触范围 x[{rmin[0]:.1f},{rmax[0]:.1f}] y[{rmin[1]:.1f},{rmax[1]:.1f}] z[{rmin[2]:.1f},{rmax[2]:.1f}]")
                
                # 3D可视化更新（在主线程中）
                if shared_data.viewer is not None:
                    # 更新夹爪平面（根据当前夹爪距离）
                    update_gripper_planes(shared_data.viewer, shared_data.gripper_mm)
                    
                    # 显示预计算的物体模型和接触面（只需要渲染，不需要重新计算）
                    if shared_data.transformed_model is not None and shared_data.model_colors is not None:
                        shared_data.viewer.update3d_pointcloud("object_model", shared_data.transformed_model, shared_data.model_colors)
                        
                        # 可视化预计算的接触面
                        if shared_data.left_contact_points is not None and shared_data.left_contact_colors is not None and shared_data.left_contact_points.shape[0] > 0:
                            shared_data.viewer.update3d_pointcloud("left_contact", shared_data.left_contact_points, shared_data.left_contact_colors)
                        
                        if shared_data.right_contact_points is not None and shared_data.right_contact_colors is not None and shared_data.right_contact_points.shape[0] > 0:
                            shared_data.viewer.update3d_pointcloud("right_contact", shared_data.right_contact_points, shared_data.right_contact_colors)
                    
                    # 运行更新循环
                    if not shared_data.viewer.update():
                        print("[可视化] 可视化窗口已关闭，进入夹爪控制")
                        try:
                            # 安全关闭3D窗口
                            shared_data.viewer.running = False
                            time.sleep(0.1)
                            shared_data.viewer.close()
                        except Exception as e:
                            print(f"[可视化] 关闭3D窗口失败: {e}")
                        shared_data.viewer = None
                        shared_data.state = FlowState.GRIPPER_CONTROL
                
                time.sleep(0.033)  # 30Hz

            elif current_state == FlowState.GRIPPER_CONTROL:
                # 检查状态是否已经被改变（避免竞态条件）
                with shared_data.lock:
                    if getattr(shared_data, 'restart_requested', False):
                        print("[夹爪控制] 检测到全局重启请求，跳过2D流程本轮")
                        continue
                    if shared_data.state != FlowState.GRIPPER_CONTROL:
                        continue  # 状态已经被改变，重新检查新状态
 
                # 若有2D关闭请求，则在主线程安全关闭
                if getattr(shared_data, 'viewer_close_requested', False):
                    shared_data.viewer_close_requested = False
                    v = shared_data.viewer
                    if v is not None:
                        try:
                            print("[触觉] 主线程关闭2D窗口...")
                            v.running = False
                            v.close()
                        except Exception as e:
                            print(f"[触觉] 关闭2D窗口出错: {e}")
                        shared_data.viewer = None
                        time.sleep(0.05)

                # 初始化触觉传感器（仅一次）
                with shared_data.lock:
                    if getattr(shared_data, 'restart_requested', False):
                        print("[夹爪控制] 重启中，跳过触觉初始化")
                        continue
                    if shared_data.tactile_left is None and shared_data.state == FlowState.GRIPPER_CONTROL:
                        try:
                            shared_data.tactile_left = TactileSensor(sensor_serial="OG000340")
                            shared_data.tactile_left.setup(reset_reference=True)
                            print("[触觉][左] 初始化成功")
                        except Exception as e:
                            print(f"[触觉][左] 初始化失败: {e}")
                            shared_data.tactile_left = None
                    if shared_data.tactile_right is None and shared_data.state == FlowState.GRIPPER_CONTROL:
                        try:
                            shared_data.tactile_right = TactileSensor(sensor_serial="OG000229")
                            shared_data.tactile_right.setup(reset_reference=True)
                            print("[触觉][右] 初始化成功")
                        except Exception as e:
                            print(f"[触觉][右] 初始化失败: {e}")
                            shared_data.tactile_right = None
                
                # 初始化2D可视化界面（仅一次）
                with shared_data.lock:
                    if getattr(shared_data, 'restart_requested', False):
                        print("[夹爪控制] 重启中，跳过2D窗口创建")
                        continue
                    if shared_data.viewer is None and shared_data.state == FlowState.GRIPPER_CONTROL:
                        print("[触觉] 正在创建2D可视化窗口...")
                        shared_data.viewer = Viewer(
                            window_title="触觉传感器",
                            window_width=1600,
                            window_height=1000,
                            num_2d_panels=4,  # 2x2布局
                            o3d_ratio=0.65,
                        )
                        if not shared_data.viewer.setup():
                            print("[触觉] 2D可视化创建失败")
                            shared_data.viewer = None
                        else:
                            # 显式打开 2D 窗口（避免闪烁）
                            shared_data.viewer.open_2d(num_2d_panels=4)
                            print("[触觉] 2D可视化窗口已打开")
                
                # 2D触觉图像显示（在主线程中）
                # 再次检查状态，避免在状态切换时继续显示
                with shared_data.lock:
                    if getattr(shared_data, 'restart_requested', False):
                        print("[夹爪控制] 重启中，停止2D显示")
                        continue
                    current_state_check = shared_data.state
                
                if current_state_check != FlowState.GRIPPER_CONTROL:
                    # 状态已改变，立即清理并跳出当前处理
                    print(f"[夹爪控制] 状态已从GRIPPER_CONTROL切换到{current_state_check.name}，停止2D显示")
                    continue
                
                viewer = shared_data.viewer
                if viewer is not None and viewer.running:
                    try:
                        # 快速获取数据（短时间持锁）
                        with shared_data.lock:
                            if shared_data.state != FlowState.GRIPPER_CONTROL:
                                continue  # 状态在获取数据时被改变
                            left_rectify = shared_data.left_rectify
                            right_rectify = shared_data.right_rectify
                            left_difference = shared_data.left_difference
                            right_difference = shared_data.right_difference
                        
                        # 显示四块触觉图（2x2布局）
                        if (left_rectify is not None and 
                            right_rectify is not None and
                            left_difference is not None and 
                            right_difference is not None):
                            
                            viewer.update2d(0, left_rectify, cmap='gray', title='Left Rectify')
                            viewer.update2d(1, right_rectify, cmap='gray', title='Right Rectify')
                            viewer.update2d(2, left_difference, cmap='gray', title='Left Difference')
                            viewer.update2d(3, right_difference, cmap='gray', title='Right Difference')
                            
                            # 驱动2D窗口刷新
                            viewer.update()
                        else:
                            # 显示占位图像
                            placeholder = np.zeros((240, 320), dtype=np.uint8)
                            viewer.update2d(0, placeholder, cmap='gray', title='Left Rectify (No Data)')
                            viewer.update2d(1, placeholder, cmap='gray', title='Right Rectify (No Data)')
                            viewer.update2d(2, placeholder, cmap='gray', title='Left Difference (No Data)')
                            viewer.update2d(3, placeholder, cmap='gray', title='Right Difference (No Data)')
                            viewer.update()
                    except Exception as e:
                        # 如果viewer在更新过程中被关闭，忽略错误
                        pass
                
                time.sleep(0.033)  # 30Hz

            elif current_state == FlowState.SHUTDOWN:
                print("[提示] 正在退出...")
                break

            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("[提示] 捕获到 Ctrl+C，正在退出...")
    finally:
        # 设置关闭标志
        shared_data.shutdown = True
        
        try:
            tracker.stop()
        except Exception:
            pass
        
        # 等待线程结束
        if keyboard_thread.is_alive():
            keyboard_thread.join(timeout=1.0)
        if tactile_thread.is_alive():
            tactile_thread.join(timeout=1.0)
        if data_thread.is_alive():
            data_thread.join(timeout=1.0)
        
        # 清理所有资源
        try:
            shared_data.cleanup_all()
            print("[系统] 所有资源已清理")
        except Exception as e:
            print(f"[系统] 资源清理时出错: {e}")


if __name__ == "__main__":
    main()


