"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例， 其他相机数据读取可能需要对应修改。 """

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os
from piper_sdk import *

count = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 使用当前文件所在目录的绝对路径，确保在任何位置运行都能正确保存
current_dir = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(current_dir, "collect_data/100data/")

# 确保保存目录存在
os.makedirs(image_save_path, exist_ok=True)
print(f"数据将保存至: {image_save_path}")

piper = C_PiperInterface_V2(dh_is_offset=1)
piper.ConnectPort()
# 使用前需要使能
piper.EnableFkCal()
# 注意，由于计算在单一线程中十分耗费资源，打开后会导致cpu占用率上升接近一倍


def data_collect():
    global count
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                           cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

        k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        if k == ord('s'):  # 键盘按一下s, 保存当前照片和机械臂位姿
            print(f"采集第{count}组数据...")
            pose = piper.GetFK('feedback')[-1]  # 获取当前机械臂状态 需要根据实际使用的机械臂获得
            print(f"机械臂pose:{pose}")

            with open(f'{image_save_path}poses.txt', 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)

            cv2.imwrite(image_save_path + str(count) + '.jpg', color_image)
            count += 1


if __name__ == "__main__":
    data_collect()