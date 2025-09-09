import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from piper_sdk.interface import C_PiperInterface_V2


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _mdeg_to_rad(angle_mdeg: int) -> float:
    """将毫度（0.001 度单位）转换为弧度。"""
    return (angle_mdeg * 1e-3) * math.pi / 180.0


def _mdeg_to_deg(angle_mdeg: int) -> float:
    """将毫度（0.001 度单位）转换为度。"""
    return angle_mdeg * 1e-3


def _deg_to_rad(angle_deg: float) -> float:
    """将度转换为弧度。"""
    return angle_deg * math.pi / 180.0


def _um_to_mm(um: int) -> float:
    """将微米（µm）转换为毫米（mm）。"""
    return um * 1e-3


def _mm001_to_mm(mm001: int) -> float:
    """将 0.001 mm 单位的整数转换为 mm。"""
    return mm001 * 1e-3


def _mdeg_s_to_rad_s(mdeg_s: int) -> float:
    """将角速度（0.001 度/秒）转换为弧度/秒。"""
    return _deg_to_rad(mdeg_s * 1e-3)


class PiperSlaveArmReader:
    """Piper V2 从臂（只读）数据读取器。

    仅进行连接与读取，不进行任何控制或使能操作，适用于主从系统中的从臂数据采集。

    功能：
    - 连接 CAN 端口（不使能）
    - 读取高频反馈（位置、速度、电流等）
    - 读取控制目标（主臂对从臂的控制指令）
    - 读取夹爪反馈与控制目标

    单位与约定：
    - 关节位置反馈来自 HighSpdInfo，单位为毫度（0.001 度）。本类默认对外提供弧度。
    - 关节速度反馈单位为 0.001 度/秒，本类默认对外提供弧度/秒。
    - 夹爪反馈单位为微米（µm），本类默认对外提供毫米（mm）。
    - 控制目标中的关节采用毫度（0.001 度）表示；本类默认对外提供弧度。
    - 控制目标中的夹爪采用 0.001 mm 表示；本类默认对外提供毫米（mm）。
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
    ) -> None:
        """创建只读从臂读取器。"""
        self._piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=judge_flag,
            can_auto_init=can_auto_init,
            dh_is_offset=dh_is_offset,
            start_sdk_joint_limit=start_sdk_joint_limit,
            start_sdk_gripper_limit=start_sdk_gripper_limit,
        )

    # ----------------------------
    # 连接 / 断开
    # ----------------------------
    def connect(self) -> None:
        """连接 CAN 端口（不调用使能）。"""
        self._piper.ConnectPort()
        time.sleep(0.1)

    def disconnect(self) -> None:
        """断开 CAN 端口（如果 SDK 暴露了断开函数，可在此调用）。"""
        # SDK 暂未直接在示例中展示 DisconnectPort，这里预留接口。
        try:
            # 若存在：self._piper.DisconnectPort()
            pass
        except Exception:
            pass

    # ----------------------------
    # 读取反馈（位置 / 速度 / 夹爪）
    # ----------------------------
    def read_positions(
        self,
        *,
        return_radians: bool = True,
    ) -> List[float]:
        """读取 7 自由度位姿 [j1..j6, gripper]。

        - return_radians：True 则关节以弧度返回；False 则以度返回。
        - 夹爪返回毫米（mm）。
        """
        highspd = self._piper.GetArmHighSpdInfoMsgs()
        gripper_fdb = self._piper.GetArmGripperMsgs()

        joint_mdeg = [
            highspd.motor_1.pos,
            highspd.motor_2.pos,
            highspd.motor_3.pos,
            highspd.motor_4.pos,
            highspd.motor_5.pos,
            highspd.motor_6.pos,
        ]
        if return_radians:
            joints = [_mdeg_to_rad(v) for v in joint_mdeg]
        else:
            joints = [_mdeg_to_deg(v) for v in joint_mdeg]

        gripper_value = float(_um_to_mm(gripper_fdb.gripper_state.grippers_angle))

        return [
            joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], gripper_value
        ]

    def read_speeds(
        self,
        *,
        return_rad_per_s: bool = True,
    ) -> List[float]:
        """读取 6 个关节角速度 [j1..j6]。

        - return_rad_per_s：True 则以弧度/秒返回；False 则以度/秒返回。
        """
        highspd = self._piper.GetArmHighSpdInfoMsgs()
        speed_mdeg_s = [
            highspd.motor_1.motor_speed,
            highspd.motor_2.motor_speed,
            highspd.motor_3.motor_speed,
            highspd.motor_4.motor_speed,
            highspd.motor_5.motor_speed,
            highspd.motor_6.motor_speed,
        ]
        if return_rad_per_s:
            return [_mdeg_s_to_rad_s(v) for v in speed_mdeg_s]
        else:
            return [v * 1e-3 for v in speed_mdeg_s]

    # ----------------------------
    # 读取控制目标（来自主臂发送给从臂的控制）
    # ----------------------------
    def read_control_targets(
        self,
        *,
        return_radians: bool = True,
    ) -> List[float]:
        """读取 7 自由度控制目标 [j1..j6, gripper]。

        - return_radians：True 则关节以弧度返回；False 则以度返回。
        - 夹爪返回毫米（mm）。
        """
        joint_ctrl = self._piper.GetArmJointCtrl()
        gripper_ctrl = self._piper.GetArmGripperCtrl()

        joints_mdeg = [
            joint_ctrl.joint_ctrl.joint_1,
            joint_ctrl.joint_ctrl.joint_2,
            joint_ctrl.joint_ctrl.joint_3,
            joint_ctrl.joint_ctrl.joint_4,
            joint_ctrl.joint_ctrl.joint_5,
            joint_ctrl.joint_ctrl.joint_6,
        ]
        if return_radians:
            joints = [_mdeg_to_rad(v) for v in joints_mdeg]
        else:
            joints = [_mdeg_to_deg(v) for v in joints_mdeg]

        gripper_value = float(_mm001_to_mm(gripper_ctrl.gripper_ctrl.grippers_angle))

        return [
            joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], gripper_value
        ]

    # ----------------------------
    # 快照（一次性读取所有常用数据）
    # ----------------------------
    def get_snapshot(
        self,
        *,
        return_radians: bool = True,
    ) -> Dict[str, object]:
        """获取一次性快照数据，便于记录或显示。

        返回字典包含：
        - joint_pos: List[float] 关节位置（6）
        - gripper: float 夹爪值（mm）
        - joint_speed: List[float] 关节速度（6）
        - joint_ctrl: List[float] 控制目标关节（6）
        - gripper_ctrl: float 控制目标夹爪（mm）
        - fps_highspd: float 高速反馈帧率（Hz）
        - fps_gripper: float 夹爪反馈帧率（Hz）
        """
        highspd = self._piper.GetArmHighSpdInfoMsgs()
        gripper_fdb = self._piper.GetArmGripperMsgs()

        joint_pos = self.read_positions(return_radians=return_radians)
        gripper_value = joint_pos[-1]
        joint_pos_only = joint_pos[:6]

        joint_speed = self.read_speeds(return_rad_per_s=return_radians)

        ctrl = self.read_control_targets(return_radians=return_radians)
        gripper_ctrl = ctrl[-1]
        joint_ctrl_only = ctrl[:6]

        return {
            "joint_pos": joint_pos_only,
            "gripper": gripper_value,
            "joint_speed": joint_speed,
            "joint_ctrl": joint_ctrl_only,
            "gripper_ctrl": gripper_ctrl,
            "fps_highspd": float(getattr(highspd, "Hz", 0.0)),
            "fps_gripper": float(getattr(gripper_fdb, "Hz", 0.0)),
        }

    # ----------------------------
    # 工具 / 访问器
    # ----------------------------
    @property
    def sdk(self) -> C_PiperInterface_V2:
        """暴露底层 SDK 实例，便于高级用法。"""
        return self._piper