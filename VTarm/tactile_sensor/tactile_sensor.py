import sys
import numpy as np
import open3d as o3d 
from xensesdk import Sensor
import time
from typing import Optional

# --- 关键配置：传感器的物理尺寸 (由客服确认的精确值) ---
SENSOR_WIDTH_MM = 19.4  # 传感器的物理宽度为19.4mm (对应400像素)
SENSOR_HEIGHT_MM = 30.8 # 传感器的物理高度为30.8mm (对应700像素)
import numpy as np
from xensesdk import Sensor
from typing import Optional


def get_depth_image(sensor: Sensor, 
                   flip_lr: bool = False, 
                   flip_ud: bool = False, 
                   reverse_depth: bool = False) -> Optional[np.ndarray]:
    """获取深度图像数据
    
    Args:
        sensor: XenseTactile传感器对象
        flip_lr: 是否左右翻转
        flip_ud: 是否上下翻转
        reverse_depth: 是否深度反向（深度值取反）
        
    Returns:
        深度图像数组，如果获取失败返回None
    """
    depth = sensor.selectSensorInfo(Sensor.OutputType.Depth)
    
    if depth is None:
        return None
    
    # 应用翻转操作
    if flip_lr:
        depth = np.fliplr(depth)
    if flip_ud:
        depth = np.flipud(depth)
    
    # 深度反向
    if reverse_depth:
        # 找到最大深度值，然后用最大值减去当前值实现反向
        max_depth = np.max(depth)
        depth = max_depth - depth
    
    return depth


def get_rectify_image(sensor: Sensor, 
                     flip_lr: bool = False, 
                     flip_ud: bool = False) -> Optional[np.ndarray]:
    """获取原始图像（Rectify）
    
    Args:
        sensor: XenseTactile传感器对象
        flip_lr: 是否左右翻转
        flip_ud: 是否上下翻转
        
    Returns:
        原始图像数组，如果获取失败返回None
    """
    rectify_img = sensor.selectSensorInfo(Sensor.OutputType.Rectify)
    
    if rectify_img is None:
        return None
    
    # 应用翻转操作
    if flip_lr:
        rectify_img = np.fliplr(rectify_img)
    if flip_ud:
        rectify_img = np.flipud(rectify_img)
    
    return rectify_img


def get_difference_image(sensor: Sensor, 
                        flip_lr: bool = False, 
                        flip_ud: bool = False) -> Optional[np.ndarray]:
    """获取差分图像（Difference）
    
    Args:
        sensor: XenseTactile传感器对象
        flip_lr: 是否左右翻转
        flip_ud: 是否上下翻转
        
    Returns:
        差分图像数组，如果获取失败返回None
    """
    diff_img = sensor.selectSensorInfo(Sensor.OutputType.Difference)
    
    if diff_img is None:
        return None
    
    # 应用翻转操作
    if flip_lr:
        diff_img = np.fliplr(diff_img)
    if flip_ud:
        diff_img = np.flipud(diff_img)
    
    return diff_img


class TactileSensor:
    """简易触觉传感器包装类。

    - 负责创建/释放 xensesdk.Sensor
    - 提供获取 Depth / Rectify / Difference 三类图像的便捷方法
    - 支持左右/上下翻转；Depth 额外支持反向
    """

    def __init__(self, sensor_serial: str = "OG000229") -> None:
        self.sensor_serial = sensor_serial
        self.sensor: Optional[Sensor] = None

    def setup(self, reset_reference: bool = True) -> None:
        """初始化并可选重置参考图像。"""
        self.sensor = Sensor.create(self.sensor_serial)
        if reset_reference and self.sensor is not None:
            try:
                self.sensor.resetReferenceImage()
            except Exception:
                pass

    def close(self) -> None:
        """释放底层资源。"""
        if self.sensor is not None:
            try:
                self.sensor.release()
            except Exception:
                pass
            self.sensor = None

    # ----------------------------
    # 数据获取便捷方法
    # ----------------------------
    def depth(self, *, flip_lr: bool = False, flip_ud: bool = False, reverse_depth: bool = False) -> Optional[np.ndarray]:
        assert self.sensor is not None, "请先调用 setup() 初始化传感器"
        return get_depth_image(self.sensor, flip_lr=flip_lr, flip_ud=flip_ud, reverse_depth=reverse_depth)

    def rectify(self, *, flip_lr: bool = False, flip_ud: bool = False) -> Optional[np.ndarray]:
        assert self.sensor is not None, "请先调用 setup() 初始化传感器"
        return get_rectify_image(self.sensor, flip_lr=flip_lr, flip_ud=flip_ud)

    def difference(self, *, flip_lr: bool = False, flip_ud: bool = False) -> Optional[np.ndarray]:
        assert self.sensor is not None, "请先调用 setup() 初始化传感器"
        return get_difference_image(self.sensor, flip_lr=flip_lr, flip_ud=flip_ud)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization import Viewer
    import time
    import numpy as np
    
    
    def _to_uint8_silent(img: np.ndarray, is_depth: bool = False) -> np.ndarray:
        """静默版本的图像转换（不打印调试信息）"""
        if img is None:
            return np.zeros((240, 320), dtype=np.uint8)
        
        if img.dtype == np.uint8:
            # uint8图像直接返回
            return img
        
        # 只有depth需要转换
        if is_depth:
            img_float = img.astype(np.float32)
            mn, mx = float(np.min(img_float)), float(np.max(img_float))
            
            if mx - mn < 1e-6:
                return np.zeros_like(img_float, dtype=np.uint8)
            
            # 深度映射：值越大（越深）-> 颜色越深（值越小）
            norm = (img_float - mn) / (mx - mn)
            return (255 - norm * 255).clip(0, 255).astype(np.uint8)
        
        return img
    
    print("[触觉传感器测试] 正在初始化传感器...")
    sensor = TactileSensor(sensor_serial="OG000229")
    sensor.setup(reset_reference=True)
    print("[触觉传感器测试] 传感器初始化完成")
    
    # 创建Viewer实例
    viewer = Viewer(
        window_title="触觉传感器测试",
        window_width=1200,
        window_height=800,
        num_2d_panels=3,  # 3个面板：depth, rectify, difference
        o3d_ratio=0.0,    # 不使用3D显示
    )
    
    if not viewer.setup():
        print("[触觉传感器测试] 可视化界面初始化失败")
        exit(1)
    
    # 打开2D窗口
    viewer.open_2d(num_2d_panels=3)
    print("[触觉传感器测试] 可视化界面已打开，按ESC退出")
    
    # 添加调试标志
    debug_printed = False
    
    try:
        # 实时可视化depth、rectify、difference三张图像
        while True:
            try:
                # 获取三张图像
                depth = sensor.depth()
                rectify = sensor.rectify()
                difference = sensor.difference()
                
                # 处理图像数据
                if not debug_printed:
                    print("\n=== 第一次数据获取，打印详细信息 ===")
                    print(f"[depth] 原始数据类型: {depth.dtype if depth is not None else 'None'}")
                    print(f"[depth] 原始数据形状: {depth.shape if depth is not None else 'None'}")
                    print(f"[depth] 原始数据范围: [{np.min(depth):.3f}, {np.max(depth):.3f}]" if depth is not None else "None")
                    print(f"[rectify] 原始数据类型: {rectify.dtype if rectify is not None else 'None'}")
                    print(f"[rectify] 原始数据形状: {rectify.shape if rectify is not None else 'None'}")
                    print(f"[rectify] 原始数据范围: [{np.min(rectify):.3f}, {np.max(rectify):.3f}]" if rectify is not None else "None")
                    print(f"[difference] 原始数据类型: {difference.dtype if difference is not None else 'None'}")
                    print(f"[difference] 原始数据形状: {difference.shape if difference is not None else 'None'}")
                    print(f"[difference] 原始数据范围: [{np.min(difference):.3f}, {np.max(difference):.3f}]" if difference is not None else "None")
                    print("=== 调试信息打印完成，后续将静默运行 ===\n")
                    debug_printed = True
                
                # 只转换depth，rectify和difference直接使用
                if depth is not None:
                    depth_u8 = _to_uint8_silent(depth, is_depth=True)
                else:
                    depth_u8 = np.zeros((240, 320), dtype=np.uint8)
                
                # rectify和difference直接使用原始数据
                rectify_img = rectify if rectify is not None else np.zeros((240, 320, 3), dtype=np.uint8)
                difference_img = difference if difference is not None else np.zeros((240, 320, 3), dtype=np.uint8)
                
                # 更新显示
                viewer.update2d(0, depth_u8, cmap='gray', title='Depth')
                viewer.update2d(1, rectify_img, cmap='gray', title='Rectify')
                viewer.update2d(2, difference_img, cmap='gray', title='Difference')
                
                # 刷新窗口
                if not viewer.update():
                    print("[触觉传感器测试] 窗口已关闭")
                    break
                    
            except KeyboardInterrupt:
                print("\n[触觉传感器测试] 用户中断")
                break
            except Exception as e:
                print(f"[触觉传感器测试] 错误: {e}")
                time.sleep(0.1)
            
            time.sleep(0.033)  # 30Hz刷新率
            
    finally:
        # 清理资源
        print("[触觉传感器测试] 正在清理资源...")
        try:
            sensor.close()
        except Exception:
            pass
        try:
            viewer.close()
        except Exception:
            pass
        print("[触觉传感器测试] 测试完成")