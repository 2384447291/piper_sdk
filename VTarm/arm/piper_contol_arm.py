import math
import time
from typing import Callable, List, Optional, Sequence, Tuple

try:
    from .interface import C_PiperInterface_V2
except ImportError:  # 兼容直接运行脚本的场景
    from piper_sdk.interface import C_PiperInterface_V2


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _rad_to_mdeg(angle_rad: float) -> int:
    """将弧度转换为毫度（0.001 度单位）。"""
    return int(round(angle_rad * 180.0 / math.pi * 1000.0))


def _mdeg_to_rad(angle_mdeg: int) -> float:
    """将毫度（0.001 度单位）转换为弧度。"""
    return (angle_mdeg * 1e-3) * math.pi / 180.0


def _mm_to_um(mm: float) -> int:
    """将毫米（mm）转换为微米（µm）。"""
    return int(round(mm * 1000.0))


def _um_to_mm(um: int) -> float:
    """将微米（µm）转换为毫米（mm）。"""
    return um * 1e-3


class PiperArmController:
    """Piper V2 机械臂（仅机械臂部分）的高层控制封装。

    功能包含：
    - 连接与使能流程
    - 关节控制模式设置
    - 发送关节与夹爪控制指令
    - 读取当前关节与夹爪状态
    - 回零（移动到零位）并带收敛判定

    单位说明：
    - 关节角反馈单位为毫度（0.001 度）。对外 API 默认返回弧度。
    - 夹爪开合反馈单位为微米（µm）。对外 API 默认返回毫米（mm），也可通过归一化函数自定义。
    """

    def __init__(
        self,
        can_name: str = "can0",
        *,
        judge_flag: bool = True,
        can_auto_init: bool = True,
        dh_is_offset: int = 1,
        start_sdk_joint_limit: bool = False,
        start_sdk_gripper_limit: bool = False,
        default_move_zero_speed: int = 30,
        default_move_policy_speed: int = 100,
        default_gripper_effort: int = 5000,
        gripper_normalize_fn: Optional[Callable[[float], float]] = None,
        gripper_denormalize_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """创建控制器实例。

        - gripper_normalize_fn(mm)->normalized：可选，将夹爪开合（单位 mm）映射为归一化数值的函数。
        - gripper_denormalize_fn(normalized)->mm：可选，上述映射的逆函数。
        """
        self._piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=judge_flag,
            can_auto_init=can_auto_init,
            dh_is_offset=dh_is_offset,
            start_sdk_joint_limit=start_sdk_joint_limit,
            start_sdk_gripper_limit=start_sdk_gripper_limit,
        )
        self._default_move_zero_speed = _clamp(default_move_zero_speed, 0, 100)
        self._default_move_policy_speed = _clamp(default_move_policy_speed, 0, 100)
        self._default_gripper_effort = int(default_gripper_effort)

        # 夹爪归一化钩子（可选）
        self._gripper_normalize_fn = gripper_normalize_fn
        self._gripper_denormalize_fn = gripper_denormalize_fn

    # ----------------------------
    # 连接 / 使能
    # ----------------------------
    def connect(self) -> None:
        """连接 CAN 端口并启动 SDK 线程。"""
        self._piper.ConnectPort()

    def enable(self, *, retry_interval_s: float = 0.01, timeout_s: Optional[float] = None) -> bool:
        """使能机械臂；成功返回 True，超时返回 False。"""
        start = time.time()
        while not self._piper.EnablePiper():
            if timeout_s is not None and (time.time() - start) > timeout_s:
                return False
            time.sleep(retry_interval_s)
        return True

    # ----------------------------
    # 状态读取
    # ----------------------------
    def get_joint_angles(self, *, in_radians: bool = True, normalize_gripper: bool = True) -> List[float]:
        """获取当前 7 自由度角度 [j1..j6, gripper]。

        - 若 in_radians=True（默认），关节角以弧度返回；否则以度返回。
        - 若 normalize_gripper=True 且提供了归一化函数，夹爪返回归一化值；否则返回毫米（mm）。
        """
        highspd = self._piper.GetArmHighSpdInfoMsgs()
        gripper_fdb = self._piper.GetArmGripperMsgs()

        # 关节角：SDK 反馈单位为毫度
        joint_mdeg = [
            highspd.motor_1.pos,
            highspd.motor_2.pos,
            highspd.motor_3.pos,
            highspd.motor_4.pos,
            highspd.motor_5.pos,
            highspd.motor_6.pos,
        ]
        if in_radians:
            joints = [_mdeg_to_rad(v) for v in joint_mdeg]
        else:
            joints = [v * 1e-3 for v in joint_mdeg]  # 度

        # 夹爪：SDK 反馈单位为微米（µm）
        gripper_mm = _um_to_mm(gripper_fdb.gripper_state.grippers_angle)
        if normalize_gripper and self._gripper_normalize_fn is not None:
            gripper_value = float(self._gripper_normalize_fn(gripper_mm))
        else:
            gripper_value = float(gripper_mm)

        return [
            joints[0],
            joints[1],
            joints[2],
            joints[3],
            joints[4],
            joints[5],
            gripper_value,
        ]

    # ----------------------------
    # 模式 / 控制
    # ----------------------------
    def set_joint_mode(
        self,
        *,
        speed_percent: Optional[int] = None,
        is_mit_mode: int = 0x00,
    ) -> None:
        """设置为 CAN 指令控制模式 + 关节运动模式。

        - speed_percent：0-100；None 时使用默认策略速度。
        - is_mit_mode：0x00 位置-速度模式；0xAD MIT 高跟随模式。
        """
        spd = _clamp(self._default_move_policy_speed if speed_percent is None else speed_percent, 0, 100)
        self._piper.ModeCtrl(0x01, 0x01, spd, is_mit_mode)

    def command_joint_and_gripper(
        self,
        joint_angles: Sequence[float],
        gripper_value: float,
        *,
        joint_in_radians: bool = True,
        gripper_is_normalized: bool = True,
        speed_percent: Optional[int] = None,
        gripper_effort: Optional[int] = None,
        small_delay_s: float = 0.001,
        enable_gripper_code: int = 0x01,

    ) -> None:
        """发送一次关节 + 夹爪控制命令。

        - joint_angles：长度为 6；当 joint_in_radians=True 时单位为弧度，否则为度。
        - gripper_value：若 gripper_is_normalized=True 且提供了反归一化函数，则视为归一化值；否则视为毫米（mm）。
        - speed_percent：可选，覆盖本次命令的速度比例。
        - gripper_effort：可选，夹爪力矩（0-5000 对应 0-5 N·m）。
        - small_delay_s：两帧之间的微小延时，避免拥塞。
        """
        if len(joint_angles) != 6:
            raise ValueError("joint_angles 长度必须为 6（对应 j1..j6）")

        # 关节单位转换

        # 关节单位转换
        if joint_in_radians:
            joints_mdeg = [_rad_to_mdeg(a) for a in joint_angles]
        else:
            joints_mdeg = [int(round(a * 1000.0)) for a in joint_angles]  # 度 -> 毫度

        # 夹爪单位转换（转为 µm）
        if gripper_is_normalized and self._gripper_denormalize_fn is not None:
            gripper_mm = float(self._gripper_denormalize_fn(gripper_value))
        else:
            gripper_mm = float(gripper_value)
        gripper_um = _mm_to_um(gripper_mm)

        # 发送关节控制帧
        self._piper.JointCtrl(
            int(joints_mdeg[0]),
            int(joints_mdeg[1]),
            int(joints_mdeg[2]),
            int(joints_mdeg[3]),
            int(joints_mdeg[4]),
            int(joints_mdeg[5]),
        )
        time.sleep(small_delay_s)

        # 发送夹爪控制帧
        effort = int(self._default_gripper_effort if gripper_effort is None else gripper_effort)
        self._piper.GripperCtrl(abs(int(gripper_um)), effort, enable_gripper_code, 0)

    def command_joints(
        self,
        joint_angles: Sequence[float],
        *,
        joint_in_radians: bool = True,
        small_delay_s: float = 0.001,
    ) -> None:
        """发送关节控制指令。

        - joint_angles：长度为 6；当 joint_in_radians=True 时单位为弧度，否则为度。
        - small_delay_s：发送后的延时。
        """
        if len(joint_angles) != 6:
            raise ValueError("joint_angles 长度必须为 6（对应 j1..j6）")

        # 关节单位转换
        if joint_in_radians:
            joints_mdeg = [_rad_to_mdeg(a) for a in joint_angles]
        else:
            joints_mdeg = [int(round(a * 1000.0)) for a in joint_angles]  # 度 -> 毫度

        # 发送关节控制帧
        self._piper.JointCtrl(
            int(joints_mdeg[0]),
            int(joints_mdeg[1]),
            int(joints_mdeg[2]),
            int(joints_mdeg[3]),
            int(joints_mdeg[4]),
            int(joints_mdeg[5]),
        )
        time.sleep(small_delay_s)

    def command_gripper(
        self,
        gripper_value: float,
        *,
        gripper_is_normalized: bool = True,
        gripper_effort: Optional[int] = None,
        enable_gripper_code: int = 0x01,
        small_delay_s: float = 0.001,
    ) -> None:
        """发送夹爪控制指令。

        - gripper_value：若 gripper_is_normalized=True 且提供了反归一化函数，则视为归一化值；否则视为毫米（mm）。
        - gripper_effort：可选，夹爪力矩（0-5000 对应 0-5 N·m）。
        - enable_gripper_code：夹爪使能代码。
        - small_delay_s：发送后的延时。
        """
        # 夹爪单位转换（转为 µm）
        if gripper_is_normalized and self._gripper_denormalize_fn is not None:
            gripper_mm = float(self._gripper_denormalize_fn(gripper_value))
        else:
            gripper_mm = float(gripper_value)
        gripper_um = _mm_to_um(gripper_mm)

        # 发送夹爪控制帧
        effort = int(self._default_gripper_effort if gripper_effort is None else gripper_effort)
        self._piper.GripperCtrl(abs(int(gripper_um)), effort, enable_gripper_code, 0)
        time.sleep(small_delay_s)

    # ----------------------------
    # 回零（移动到零位）
    # ----------------------------
    def move_to_zero(
        self,
        target_joint_angles: Sequence[float],
        target_gripper_value: float,
        *,
        joint_in_radians: bool = True,
        gripper_is_normalized: bool = True,
        speed_percent: Optional[int] = None,
        iterations: int = 10,
        iteration_interval_s: float = 0.1,
    ) -> None:
        """向零位（或给定目标位姿）重复发送指令。

        参考官方示例：先连续发送多帧，确保控制器锁定目标。
        """
        if len(target_joint_angles) != 6:
            raise ValueError("target_joint_angles 长度必须为 6（对应 j1..j6）")

        for _ in range(max(1, iterations)):
            self.command_joint_and_gripper(
                target_joint_angles,
                target_gripper_value,
                joint_in_radians=joint_in_radians,
                gripper_is_normalized=gripper_is_normalized,
                speed_percent=speed_percent if speed_percent is not None else self._default_move_zero_speed,
            )
            time.sleep(iteration_interval_s)

    def wait_until_reached(
        self,
        target_joint_angles: Sequence[float],
        target_gripper_value: float,
        *,
        joint_in_radians: bool = True,
        gripper_is_normalized: bool = True,
        tolerance: float = 0.2,
        timeout_s: float = 8.0,
        check_interval_s: float = 0.5,
    ) -> bool:
        """轮询反馈，直到到达目标（误差阈值内）或超时。

        - tolerance：当 joint_in_radians=True 时为弧度，否则为度；此处不检查夹爪误差。
        若到达返回 True；超时返回 False。
        """
        start = time.time()

        # 目标准备
        if joint_in_radians:
            target = list(target_joint_angles)
        else:
            target = [math.radians(d) for d in target_joint_angles]

        while True:
            # 读取当前
            current = self.get_joint_angles(in_radians=True, normalize_gripper=gripper_is_normalized)
            current_joints = current[:6]

            # 比较误差
            max_abs_err = max(abs(c - t) for c, t in zip(current_joints, target))
            if max_abs_err <= tolerance:
                # 额外等待时间，确保稳定
                time.sleep(2.0)
                return True

            if (time.time() - start) > timeout_s:
                return False

            time.sleep(check_interval_s)

    # ----------------------------
    # 工具 / 访问器
    # ----------------------------
    @property
    def sdk(self) -> C_PiperInterface_V2:
        """暴露底层 SDK 实例，便于高级用法。"""
        return self._piper
