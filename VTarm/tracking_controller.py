import time
import logging
import numpy as np

from shm_lib.pubsub_manager import PubSubManager
from shm_lib.shared_memory_util import encode_text_prompt


CONTROL_TOPIC = "tracking_control"
POSE_TOPIC = "tracking_poses"


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)
    return R


class TrackingController:
    def __init__(self, port: int = 10000, authkey: bytes = b"foundationpose") -> None:
        self.port = port
        self.authkey = authkey
        self.pubsub = PubSubManager(port=self.port, authkey=self.authkey)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        logging.debug("[Tracking] 启动 PubSub 管理器")
        self.pubsub.start(role="both")

        topics_config = {
            CONTROL_TOPIC: {
                "examples": {
                    "command": encode_text_prompt("start"),
                    "object_prompt": np.zeros(256, dtype=np.uint8),
                    "object_name": encode_text_prompt("default_object"),
                },
                "buffer_size": 5,
                "mode": "consumer",
            },
            POSE_TOPIC: {
                "examples": {
                    "pose_matrix": np.eye(4, dtype=np.float32),
                    "timestamp": np.float64(0.0),
                    "frame_idx": np.int32(0),
                    "object_name": encode_text_prompt("default_object"),
                },
                "buffer_size": 50,
                "mode": "broadcast",
            },
        }

        self.pubsub.setup_subscriber(topics_config)
        for topic_name, config in topics_config.items():
            self.pubsub.create_topic(
                topic_name, config["examples"], config["buffer_size"], config["mode"]
            )

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        logging.debug("[Tracking] 停止 PubSub 管理器")
        self.pubsub.stop(role="both")
        self._started = False

    def send_start(self, object_prompt: str, object_name: str) -> bool:
        command_data = {
            "command": encode_text_prompt("start"),
            "object_prompt": encode_text_prompt(object_prompt),
            "object_name": encode_text_prompt(object_name),
        }
        ok = self.pubsub.publish(CONTROL_TOPIC, command_data)
        logging.info(
            f"[Tracking] 发送开始: {object_name}, prompt='{object_prompt}', ok={ok}"
        )
        return ok

    def send_stop(self) -> bool:
        command_data = {
            "command": encode_text_prompt("stop"),
            "object_prompt": np.zeros(256, dtype=np.uint8),
            "object_name": encode_text_prompt(""),
        }
        ok = self.pubsub.publish(CONTROL_TOPIC, command_data)
        logging.info(f"[Tracking] 发送停止: ok={ok}")
        return ok

    def get_latest_pose(self):
        try:
            data = self.pubsub.get_latest_data(POSE_TOPIC)
            if data is None:
                return None
            return data
        except Exception:
            return None

    def request_average_pose(self, expected_samples: int = 10, wait_timeout: float = 20.0):
        logging.info("[Tracking] 等待 buffer 中的最终位姿样本...")
        translations = []
        quaternions = []
        start_time = time.time()
        
        # 先停止，触发 client 发布最终姿态
        self.send_stop()
        
        # 等待下位机开始发布最终位姿
        logging.info("[Tracking] 等待下位机发布最终位姿...")
        time.sleep(0.5)  # 给下位机更多时间
        
        # 记录停止命令发送后的时间戳，用于过滤数据
        stop_command_time = time.time()
        
        # 使用时间戳而不是frame_idx来过滤数据，更可靠
        last_processed_timestamp = 0.0
        
        while (
            len(translations) < expected_samples
            and (time.time() - start_time) < wait_timeout
        ):
            try:
                data = self.pubsub.get_latest_data(POSE_TOPIC)
                if data is not None:
                    data_timestamp = data.get("timestamp", 0.0)
                    
                    # 只处理停止命令后发布的新数据（基于时间戳）
                    if data_timestamp > last_processed_timestamp and data_timestamp >= stop_command_time:
                        last_processed_timestamp = data_timestamp
                        pose = data["pose_matrix"]
                        R = pose[:3, :3].astype(np.float64)
                        t = pose[:3, 3].astype(np.float64)
                        translations.append(t)

                        q = rotation_matrix_to_quaternion(R)
                        if len(quaternions) > 0 and np.dot(q, quaternions[-1]) < 0:
                            q = -q
                        quaternions.append(q)

                        logging.info(
                            f"[Tracking] 已收集 {len(translations)}/{expected_samples} 样本 (timestamp: {data_timestamp:.3f})"
                        )
                    else:
                        # 跳过旧数据或停止前的数据
                        if data_timestamp < stop_command_time:
                            logging.debug(f"[Tracking] 跳过停止前的数据: timestamp={data_timestamp:.3f} < {stop_command_time:.3f}")
                        
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"[Tracking] 读取位姿出错: {e}")
                time.sleep(0.1)

        if len(translations) == 0:
            logging.warning("[Tracking] 未收集到任何位姿样本")
            return None

        t_avg = np.mean(np.stack(translations, axis=0), axis=0)
        Q = np.stack(quaternions, axis=0)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        q_avg = np.mean(Q, axis=0)
        q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-12)
        R_avg = quaternion_to_rotation_matrix(q_avg)

        T_avg = np.eye(4, dtype=np.float32)
        T_avg[:3, :3] = R_avg.astype(np.float32)
        T_avg[:3, 3] = t_avg.astype(np.float32)

        logging.info(
            f"[Tracking] 计算完成平均位姿，共 {len(translations)} 个样本"
        )
        return T_avg


