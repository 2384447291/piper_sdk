#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时可视化机械臂 FK feedback（末端位姿）的简单工具

假设 feedback 格式为一个长度 6 的 list/array:
    [x, y, z, rx, ry, rz]
其中 x,y,z 单位可为 mm 或 m，rx,ry,rz 单位可为 deg 或 rad（脚本可配置）。

功能：
- 连接 piper_sdk（如果可用），周期性读取 piper.GetFK('feedback')[-1]
- 将位姿转换为位置与旋转矩阵
- 在 matplotlib 3D 视图中实时显示末端坐标系（三轴）和一个旋转的立方体作为工具示意
- 当无法连接 piper 时提供仿真数据模式（demo）

依赖：numpy, matplotlib, (可选) scipy, piper_sdk

用法示例：
    python real_time_feedback_visualizer.py

按 Ctrl-C 退出。
"""

import time
import argparse
import sys
import math
import threading

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 尝试导入 scipy Rotation（可选），否则使用自定义函数
try:
    from scipy.spatial.transform import Rotation as Rsc
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 尝试导入 piper_sdk（如果没有则进入 demo 模式）
try:
    from piper_sdk import C_PiperInterface_V2
    _HAS_PIPER = True
except Exception:
    _HAS_PIPER = False


# -------------------- 帮助函数 --------------------

def euler_R_z_y_x(rx, ry, rz):
    """按照 R = Rz @ Ry @ Rx（先绕 X，后绕 Y，最后绕 Z）的顺序生成旋转矩阵。
    参数为弧度。
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_to_transform(feedback, units_trans='mm', units_angle='deg', euler_order='z_y_x'):
    """
    把 feedback -> 4x4 齐次变换矩阵
    units_trans: 'mm' 或 'm'
    units_angle: 'deg' 或 'rad'
    euler_order: 目前仅支持 'z_y_x' (R = Rz@Ry@Rx)；可扩展
    返回 (pos_m, R)
    """
    x, y, z, rx, ry, rz = feedback
    # 平移单位
    if units_trans == 'mm':
        pos = np.array([x / 1000.0, y / 1000.0, z / 1000.0])
    else:
        pos = np.array([x, y, z])

    # 角度单位
    if units_angle == 'deg':
        rx, ry, rz = np.deg2rad([rx, ry, rz])

    # 目前只实现 z_y_x 顺序（与你的代码一致）
    if euler_order == 'z_y_x':
        R = euler_R_z_y_x(rx, ry, rz)
    else:
        # 如果有 scipy 可用，利用 scipy 的 from_euler 更灵活
        if _HAS_SCIPY:
            # 例如 euler_order='zyx' -> seq='zyx'
            seq = euler_order.replace('_', '')
            R = Rsc.from_euler(seq, [rz, ry, rx], degrees=False).as_matrix()
        else:
            raise ValueError('unsupported euler_order and scipy not available')

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return pos, R


# -------------------- 可视化类 --------------------
class RealTimeVisualizer:
    def __init__(self, units_trans='mm', units_angle='deg', euler_order='z_y_x', scale=0.06):
        self.units_trans = units_trans
        self.units_angle = units_angle
        self.euler_order = euler_order
        self.scale = scale  # 坐标轴长度（m）

        # matplotlib 图形初始化
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._init_plot_elements()

        # 数据缓冲
        self.latest_feedback = None
        self.lock = threading.Lock()

    def _init_plot_elements(self):
        # 轴线用 3 条 Line3D 对象
        self.line_x, = self.ax.plot([], [], [], linewidth=3)
        self.line_y, = self.ax.plot([], [], [], linewidth=3)
        self.line_z, = self.ax.plot([], [], [], linewidth=3)

        # 立方体边线列表
        self.cube_lines = []
        for _ in range(12):
            l, = self.ax.plot([], [], [], color='k')
            self.cube_lines.append(l)

        # 文本显示
        self.text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes)

        # 参考世界坐标系
        self.ax.quiver(0, 0, 0, 0.05, 0, 0, color='r', arrow_length_ratio=0.2)
        self.ax.quiver(0, 0, 0, 0, 0.05, 0, color='g', arrow_length_ratio=0.2)
        self.ax.quiver(0, 0, 0, 0, 0, 0.05, color='b', arrow_length_ratio=0.2)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Real-time end-effector pose')

        # cube reference (centered at origin, edge length s)
        self.s = 0.08
        self.cube_verts = np.array([[-self.s/2, -self.s/2, -self.s/2],
                                    [ self.s/2, -self.s/2, -self.s/2],
                                    [ self.s/2,  self.s/2, -self.s/2],
                                    [-self.s/2,  self.s/2, -self.s/2],
                                    [-self.s/2, -self.s/2,  self.s/2],
                                    [ self.s/2, -self.s/2,  self.s/2],
                                    [ self.s/2,  self.s/2,  self.s/2],
                                    [-self.s/2,  self.s/2,  self.s/2]])
        self.edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    def update_from_feedback(self, feedback):
        with self.lock:
            self.latest_feedback = feedback

    def _draw_frame(self, origin, R):
        # origin: 3-vector (m), R: 3x3
        s = self.scale
        x_axis = origin + R[:, 0] * s
        y_axis = origin + R[:, 1] * s
        z_axis = origin + R[:, 2] * s

        # update lines
        self.line_x.set_data([origin[0], x_axis[0]], [origin[1], x_axis[1]])
        self.line_x.set_3d_properties([origin[2], x_axis[2]])

        self.line_y.set_data([origin[0], y_axis[0]], [origin[1], y_axis[1]])
        self.line_y.set_3d_properties([origin[2], y_axis[2]])

        self.line_z.set_data([origin[0], z_axis[0]], [origin[1], z_axis[1]])
        self.line_z.set_3d_properties([origin[2], z_axis[2]])

    def _draw_cube(self, origin, R):
        verts = (R @ self.cube_verts.T).T + origin
        for idx, (i, j) in enumerate(self.edges):
            p0 = verts[i]
            p1 = verts[j]
            self.cube_lines[idx].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            self.cube_lines[idx].set_3d_properties([p0[2], p1[2]])

    def render_once(self):
        with self.lock:
            fb = self.latest_feedback
        if fb is None:
            return
        pos, R = pose_to_transform(fb, units_trans=self.units_trans,
                                   units_angle=self.units_angle,
                                   euler_order=self.euler_order)
        self._draw_frame(pos, R)
        self._draw_cube(pos, R)

        # update text
        angs_deg = np.rad2deg([math.atan2(R[2,1], R[2,2]),
                                math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2)),
                                math.atan2(R[1,0], R[0,0])])
        txt = f"pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m\napprox euler (deg): [{angs_deg[0]:.2f}, {angs_deg[1]:.2f}, {angs_deg[2]:.2f}]"
        self.text.set_text(txt)

        # keep axes limits centered on current pos
        lim = 0.2
        self.ax.set_xlim(pos[0]-lim, pos[0]+lim)
        self.ax.set_ylim(pos[1]-lim, pos[1]+lim)
        self.ax.set_zlim(pos[2]-lim, pos[2]+lim)

        # redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# -------------------- Piper reader thread --------------------
class PiperReader(threading.Thread):
    def __init__(self, visualizer, poll_hz=50, demo=False):
        super().__init__()
        self.daemon = True
        self.viz = visualizer
        self.poll_hz = poll_hz
        self.demo = demo
        self._stop = threading.Event()

        self.piper = None
        if not demo and _HAS_PIPER:
            try:
                self.piper = C_PiperInterface_V2(dh_is_offset=1)
                self.piper.ConnectPort()
                # enable if needed
                try:
                    self.piper.EnableFkCal()
                except Exception:
                    pass
            except Exception as e:
                print('无法连接 piper SDK，进入 demo 模式:', e)
                self.demo = True

    def stop(self):
        self._stop.set()

    def run(self):
        period = 1.0 / max(1, self.poll_hz)
        t = 0
        while not self._stop.is_set():
            try:
                if self.demo or not _HAS_PIPER:
                    # 生成演示数据：绕 Z 旋转并做小振幅移动
                    ang = (time.time() * 20) % 360
                    fb = np.array([220 + 10 * math.sin(time.time()),
                                   -2 + 2 * math.sin(time.time()*0.7),
                                   221 + 5 * math.cos(time.time()*0.9),
                                   ang, 48.26947410103906, 142.95597182566064])
                else:
                    # 从 piper 读取最新的 FK
                    # 注意 piper.GetFK('feedback') 可能返回一个列表/tuple，其中最后一个元素就是角度/位姿
                    raw = self.piper.GetFK('feedback')
                    # 保证兼容性：如果返回是 list of lists
                    if isinstance(raw, (list, tuple)):
                        fb = raw[-1]
                    else:
                        fb = raw
                    fb = np.array(fb, dtype=float)

                # 在这里我们假定输入是 [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)]，若不同请修改 visualizer 配置
                self.viz.update_from_feedback(fb)
            except Exception as e:
                print('读取反馈出错:', e)
            time.sleep(period)


# -------------------- 主函数 --------------------

def main():
    parser = argparse.ArgumentParser(description='实时可视化机械臂 feedback')
    parser.add_argument('--demo', action='store_true', help='使用演示数据，不连接 piper')
    parser.add_argument('--hz', type=int, default=20, help='刷新频率 (Hz)')
    parser.add_argument('--units_trans', choices=['mm', 'm'], default='mm', help='平移单位')
    parser.add_argument('--units_angle', choices=['deg', 'rad'], default='deg', help='角度单位')
    parser.add_argument('--euler_order', default='z_y_x', help="欧拉角顺序 (默认 z_y_x 对应 R = Rz@Ry@Rx)")
    args = parser.parse_args()

    viz = RealTimeVisualizer(units_trans=args.units_trans, units_angle=args.units_angle, euler_order=args.euler_order)
    reader = PiperReader(viz, poll_hz=args.hz, demo=args.demo or (not _HAS_PIPER))
    reader.start()

    print('开始实时可视化，按 Ctrl-C 退出')
    try:
        while True:
            viz.render_once()
            time.sleep(1.0 / max(10, args.hz))
    except KeyboardInterrupt:
        print('退出...')
        reader.stop()
        reader.join()


if __name__ == '__main__':
    main()
