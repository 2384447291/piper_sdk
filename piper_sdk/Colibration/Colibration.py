# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rsc

np.set_printoptions(precision=8, suppress=True)

# 使用脚本所在目录的绝对路径，确保在任何位置运行都能找到文件
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "100data"
iamges_path = os.path.join(current_dir, f"collect_data/{file_name}")  # 手眼标定采集的标定版图片所在路径
arm_pose_file = os.path.join(current_dir, f"collect_data/{file_name}/poses.txt")  # 采集标定板图片时对应的机械臂末端的位姿


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = pose

    # 单位转换
    x = x_mm / 1000.0
    y = y_mm / 1000.0
    z = z_mm / 1000.0

    # 角度从度->弧度
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    return R, t


def camera_calibrate(iamges_path):
    print("++++++++++开始相机标定++++++++++++++")
    # 角点的个数以及棋盘格间距
    XX = 11  # 标定板的中长度对应的角点的个数
    YY = 8  # 标定板的中宽度对应的角点的个数
    L = 0.03  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    size = None      # 初始化图像尺寸变量

    # 自动检测标定数据文件夹中的图片
    image_files = [f for f in os.listdir(iamges_path) if f.endswith('.jpg') and f[:-4].isdigit()]
    image_files.sort(key=lambda x: int(x[:-4]))  # 按照数字顺序排序
    
    print(f"共找到 {len(image_files)} 张标定图片")
    
    for i, img_file in enumerate(image_files):
        image = os.path.join(iamges_path, img_file)
        print(f"正在处理第{i+1}/{len(image_files)}张图片：{image}")

        img = cv2.imread(image)
        if img is None:
            print(f"无法读取图像：{image}")
            continue
            
        print(f"图像大小： {img.shape}")
        # h_init, width_init = img.shape[:2]
        # img = cv2.resize(src=img, dsize=(width_init // 2, h_init // 2))
        # print(f"图像大小(resize)： {img.shape}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        
        if not ret or corners is None:
            print(f"未能在图像中找到 {XX}x{YY} 的棋盘格角点，跳过此图像")
            continue
            
        # 角点找到后才打印信息
        print(f"左上角点：{corners[0, 0]}")
        print(f"右下角点：{corners[-1, -1]}")

        # 绘制角点并显示图像
        cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
        cv2.imshow('Chessboard', img)

        cv2.waitKey(3000)  ## 停留3s, 观察找到的角点是否正确

        # 此时已确认找到角点
        obj_points.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

    N = len(img_points)
    
    if not img_points:
        raise ValueError("未找到有效的棋盘格角点，请检查图像路径和棋盘格参数")
    
    if size is None:
        raise ValueError("未能获取有效的图像尺寸，请检查图像是否正确加载")

    # 标定得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # print("ret:", ret)
    print("内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs


def process_arm_pose(arm_pose_file):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        R, t = pose_to_homogeneous_matrix(pose=pose)
        R_arm.append(R)
        t_arm.append(t)
    return R_arm, t_arm


def hand_eye_calibrate():
    rvecs, tvecs = camera_calibrate(iamges_path=iamges_path)
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file)

    # 确保相机和机械臂的数据数量匹配
    n_images = len(rvecs)
    n_poses = len(R_arm)
    
    if n_images != n_poses:
        print(f"警告：找到的有效图像数量({n_images})与位姿数据数量({n_poses})不匹配！")
        # 使用较小的数量
        n = min(n_images, n_poses)
        rvecs = rvecs[:n]
        tvecs = tvecs[:n]
        R_arm = R_arm[:n]
        t_arm = t_arm[:n]
        print(f"将使用前 {n} 组数据进行标定")

    # 使用TSAI方法进行手眼标定，该方法通常具有较好的精度
    # 返回值：R - 旋转矩阵，t - 平移向量（无误差评估返回值）
    R, t = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    # 计算标定结果的可靠性评估
    # 通过将标定结果应用到原始数据上，检查变换前后的一致性
    consistency_errors = []
    for i in range(len(R_arm)):
        # 根据AX=XB，计算 A_i * X 和 X * B_i 的差异
        # 其中A是相机位姿变换，B是机械臂位姿变换，X是我们求解的手眼变换
        A_transform = np.eye(4)
        A_transform[:3, :3] = cv2.Rodrigues(rvecs[i])[0]
        A_transform[:3, 3] = tvecs[i].reshape(-1)
        
        B_transform = np.eye(4)
        B_transform[:3, :3] = R_arm[i]
        B_transform[:3, 3] = t_arm[i].reshape(-1)
        
        X_transform = np.eye(4)
        X_transform[:3, :3] = R
        X_transform[:3, 3] = t.reshape(-1)
        
        # 计算 A*X 和 X*B 之间的差异
        AX = A_transform @ X_transform
        XB = X_transform @ B_transform
        
        # 计算旋转部分的差异（弗罗贝尼乌斯范数）
        rot_error = np.linalg.norm(AX[:3, :3] - XB[:3, :3], 'fro')
        # 计算平移部分的差异（欧几里得距离）
        trans_error = np.linalg.norm(AX[:3, 3] - XB[:3, 3])
        
        consistency_errors.append((rot_error, trans_error))
    
    # 计算平均误差
    avg_rot_error = np.mean([e[0] for e in consistency_errors])
    avg_trans_error = np.mean([e[1] for e in consistency_errors])
    
    print("+++++++++++手眼标定完成+++++++++++++++")
    print(f"标定一致性评估：")
    print(f"平均旋转误差: {avg_rot_error:.6f}")
    print(f"平均平移误差: {avg_trans_error:.6f} (单位与输入相同)")
    print(f"样本数量: {len(consistency_errors)}")
    
    # 如果误差过大，给出警告
    if avg_rot_error > 0.1 or avg_trans_error > 10.0:  # 阈值需要根据实际应用调整
        print("警告：标定误差较大，可能需要重新采集数据或尝试其他标定方法")
    return R, t


if __name__ == "__main__":
    R, t = hand_eye_calibrate()

    print("旋转矩阵：")
    print(R)
    print("平移向量：")
    print(t)


    rot = Rsc.from_matrix(R)
    euler_rad = rot.as_euler('xyz', degrees=False)  # rad
    euler_deg = rot.as_euler('xyz', degrees=True)   # deg

    print("欧拉角 (rad):", euler_rad)
    print("欧拉角 (deg):", euler_deg)
    # 旋转矩阵到欧拉角