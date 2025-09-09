from numbers import Rational
from piper_sdk import *
import time
import numpy as np
import open3d as o3d
from scipy.stats import linregress
import matplotlib.pyplot as plt

from arm.piper_slave_arm import PiperSlaveArmReader
from realsense.realsense import RealSense

# 相机外参（相对于主坐标系/夹爪中心）
CAMERA_POS_MM = [-103.84, 8.5, 123.6]
# 旋转顺序（世界→相机）：y轴 -28°，z轴 +90°，x轴 +90°（与 Test_gripper_single_arm.py 保持一致）
YAW_Y_DEG = -28.0
ROLL_Z_DEG = 90.0
PITCH_X_DEG = 90.0

ratio = 1.0

from tactile_sensor.tactile_sensor import (
    TactileSensorViewer,
    colors_from_height,
    SENSOR_WIDTH_MM,
    SENSOR_HEIGHT_MM,
)

def fit_ab(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    res = linregress(x, y)  # res.slope, res.intercept
    return res.slope, res.intercept



def main():
    # 初始化 piper 从臂（只读）
    print("正在初始化机械臂...")
    piper = PiperSlaveArmReader()
    piper.connect()
    print("机械臂连接成功")

    # 初始化RealSense相机
    print("正在初始化RealSense相机...")
    realsense_camera = RealSense(width=640, height=480, fps=30, use_color=False, non_blocking=True)
    print("RealSense相机初始化成功")


    # # 初始化两侧触觉传感器（仅采集，不可视化）
    # print("正在初始化触觉传感器...")
    # tactile_sensor_left = None
    # tactile_sensor_right = None
    # tactile_sensor_left = TactileSensorViewer(sensor_serial="OG000340", z_threshold_mm=0.05)
    # tactile_sensor_left.setup(visualize=False, wait_first_frame=True)
    # tactile_sensor_right = TactileSensorViewer(sensor_serial="OG000229", z_threshold_mm=0.05)
    # tactile_sensor_right.setup(visualize=False, wait_first_frame=True)
    # print("触觉传感器初始化成功")

    # 初始化可视化器
    print("正在初始化可视化界面...")
    viewer = PointCloudViewer(
        window_title="触觉传感器与RealSense点云融合显示",
        window_width=1200,
        window_height=800
    )
    
    if not viewer.setup():
        print("可视化界面初始化失败")
        return
    
    # 添加世界坐标轴（触觉传感器中心）
    viewer.add_coordinate_frame("world", size=50)

    rotation_matrix = RealSense.get_R_cam_to_world()
    # 配置 RealSense 外参并添加相机坐标轴
    RealSense.configure_extrinsics(
        pos_mm=CAMERA_POS_MM,
        yaw_y_deg=YAW_Y_DEG,
        roll_z_deg=ROLL_Z_DEG,
        pitch_x_deg=PITCH_X_DEG,
    )
    viewer.add_coordinate_frame_with_pose(
        name="camera",
        size=20,
        position=RealSense.get_camera_pos_mm().tolist(),
        rotation_matrix=RealSense.get_R_cam_to_world(),
    )
    
    
    print("可视化界面初始化成功")

    try:
        while viewer.is_running():
            # 读取手爪位置，position[6]
            position = piper.read_positions()
            pos6_val = position[6]
            distance_mm = ratio * pos6_val 
            # 两侧各取一半偏移（沿 z 方向）
            half_z = distance_mm / 2.0

            # === 获取触觉传感器点云（映射到公共坐标系）===
            # left_pts, left_z = tactile_sensor_left.get_mapped_point_cloud(
            #     is_left=True,
            #     y_plane_offset=half_z,
            #     flip_lr=True,
            #     flip_ud=True,
            #     return_z=True,
            # )
            # right_pts, right_z = tactile_sensor_right.get_mapped_point_cloud(
            #     is_left=False,
            #     y_plane_offset=half_z,
            #     flip_lr=False,
            #     flip_ud=True,
            #     return_z=True,
            # )

            # 合并触觉点云与颜色
            # tactile_points = np.vstack([left_pts, right_pts]) if (left_pts.size + right_pts.size) > 0 else np.empty((0, 3), dtype=np.float64)
            # left_colors = colors_from_height(left_z) if left_z.size > 0 else np.empty((0, 3), dtype=np.float64)
            # right_colors = colors_from_height(right_z) if right_z.size > 0 else np.empty((0, 3), dtype=np.float64)
            # tactile_colors = np.vstack([left_colors, right_colors]) if (left_colors.size + right_colors.size) > 0 else np.empty((0, 3), dtype=np.float64)

            # === 获取RealSense点云 ===
            realsense_points = np.empty((0, 3))
            realsense_colors = np.empty((0, 3))
            
            # 获取点云（RS 原生坐标，单位米），再统一转换到世界系
            rs_pts_raw, rs_colors_raw = realsense_camera.get_pointcloud(
                min_distance=0.05, max_distance=0.2
            )
            if rs_pts_raw is not None and rs_pts_raw.shape[0] > 0:
                world_pts_m = RealSense.transform_points_cam_to_world(rs_pts_raw) 
                realsense_points = world_pts_m * 1000.0
                realsense_colors = rs_colors_raw
                    

            # # === 更新可视化 ===
            # # 更新触觉传感器点云（只有在有数据时才更新）
            # if tactile_points.shape[0] > 0:
            #     viewer.update_pointcloud("tactile", tactile_points, tactile_colors)
            
            # 更新RealSense点云（只有在有数据时才更新）
            if realsense_points.shape[0] > 0:
                viewer.update_pointcloud("realsense", realsense_points, realsense_colors)
            
            # 每帧刷新触觉传感器边框（垂直于Y轴）
            H = float(SENSOR_HEIGHT_MM)
            W = float(SENSOR_WIDTH_MM)
            
            # 左侧传感器边框（Y = -half_z）
            left_corners = np.array([
                [-H/2.0, -half_z, -W/2.0],
                [ +H/2.0, -half_z, -W/2.0],
                [ +H/2.0, -half_z,  +W/2.0],
                [ -H/2.0, -half_z,  +W/2.0],
            ], dtype=np.float64)
            left_edges = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
            left_colors = np.tile([[0.2,0.8,0.2]], (4,1))  # 绿色
            viewer.add_line_set("left_border", left_corners, left_edges, left_colors)
            
            # 右侧传感器边框（Y = +half_z）
            right_corners = np.array([
                [-H/2.0, +half_z, -W/2.0],
                [ +H/2.0, +half_z, -W/2.0],
                [ +H/2.0, +half_z,  +W/2.0],
                [ -H/2.0, +half_z,  +W/2.0],
            ], dtype=np.float64)
            right_edges = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
            right_colors = np.tile([[0.8,0.2,0.2]], (4,1))  # 红色
            viewer.add_line_set("right_border", right_corners, right_edges, right_colors)
            
            # # # 运行更新循环
            if not viewer.run_update_cycle():
                break
    finally:
        print("正在清理资源...")
        
        # 关闭可视化器
        viewer.close()
        
        # 释放RealSense相机
        try:
            realsense_camera.release()
            print("RealSense相机已释放")
        except Exception as e:
            print(f"释放RealSense相机时出错: {e}")
        
        # 释放触觉传感器（当前未启用）
            
        print("资源清理完成")


if __name__ == "__main__":
    main()









