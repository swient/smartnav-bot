#!/usr/bin/env python3
"""使用者身份驗證節點"""

import threading
import numpy as np
from typing import Optional, cast
from dataclasses import dataclass

import message_filters
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from smartnav_msgs.msg import FaceEmbedding
from smartnav_msgs.srv import RegisterFace
from smartnav_brain.user_manager import UserType, UserManager
from smartnav_brain.brain_utils import compute_similarity


@dataclass
class UserRegistrationInfo:
    """使用者註冊資訊資料類別"""

    user_uuid: str
    user_name: str
    target_samples: int = 5
    collected_samples: int = 0


class UserAuthNode(Node):
    """使用者身份驗證節點"""

    def __init__(self):
        """初始化使用者身份驗證節點"""
        super().__init__("user_auth_node")

        # 宣告參數
        self.declare_parameter(
            "face_embedding_topic",
            "face_embedding",
            ParameterDescriptor(description="人臉向量話題"),
        )
        self.declare_parameter(
            "image_capture_topic",
            "image_raw",
            ParameterDescriptor(description="相機影像話題"),
        )

        # 讀取與驗證參數
        face_embedding_topic = self.get_parameter("face_embedding_topic").get_parameter_value().string_value
        image_capture_topic = self.get_parameter("image_capture_topic").get_parameter_value().string_value

        # 初始化使用者管理器
        try:
            self.user_manager = UserManager()
        except Exception as e:
            self.get_logger().error(f"使用者管理器初始化失敗: {e}")
            raise

        # 建立註冊人臉服務
        self.register_face_service = self.create_service(
            RegisterFace,
            "register_face",
            self._register_face_callback,
        )

        # 訂閱人臉向量話題
        self.face_embedding_sub = message_filters.Subscriber(
            self,
            FaceEmbedding,
            face_embedding_topic,
        )

        # 訂閱相機影像
        self.image_capture_sub = message_filters.Subscriber(
            self,
            Image,
            image_capture_topic,
        )

        self.face_image_sync = message_filters.ApproximateTimeSynchronizer(
            [self.face_embedding_sub, self.image_capture_sub],
            queue_size=10,
            slop=0.1,
        )
        self.face_image_sync.registerCallback(self._synced_face_image_callback)

        self.bridge = CvBridge()

        self.current_registration: Optional[UserRegistrationInfo] = None
        self.is_registering_face = False
        self.face_reg_timer = None
        self.face_lock = threading.Lock()

        self.get_logger().info("✓ 使用者身份驗證節點已初始化")
        self.get_logger().info(f"  訂閱話題: {face_embedding_topic}")
        self.get_logger().info(f"  訂閱話題: {image_capture_topic}")

    def _register_face_callback(self, request, response):
        """處理註冊人臉服務請求"""
        timestamp = self.get_clock().now().nanoseconds / 1e9

        with self.face_lock:
            if self.is_registering_face:
                response.success = False
                response.message = "已經在進行人臉註冊，請稍後再試"
                return response

            try:
                user_uuid = self.user_manager.register_user(
                    created_at=timestamp,
                    user_name=request.user_name,
                    user_type=UserType(request.user_type.type),
                    description=request.description,
                )

                self.current_registration = UserRegistrationInfo(
                    user_uuid=user_uuid,
                    user_name=request.user_name,
                    target_samples=request.num_samples,
                )
            except Exception as e:
                response.success = False
                response.message = f"使用者註冊失敗: {e}"
                return response

            self.is_registering_face = True

            if self.face_reg_timer:
                self.face_reg_timer.cancel()
            self.face_reg_timer = threading.Timer(20.0, self._face_registration_timeout)
            self.face_reg_timer.start()

            response.success = True
            response.message = "人臉註冊已開始，請看向攝影機"
            return response

    def _face_registration_timeout(self):
        """處理人臉註冊超時"""
        with self.face_lock:
            if self.is_registering_face:
                self.is_registering_face = False

                if self.current_registration and self.current_registration.user_uuid:
                    need_delete_uuid = self.current_registration.user_uuid

                self.current_registration = None

        if need_delete_uuid:
            try:
                self.user_manager.delete_user(need_delete_uuid)
            except Exception as e:
                self.get_logger().error(f"超時刪除使用者失敗: {e}")

        self.get_logger().warn("人臉註冊超時，請重新嘗試")

    def _synced_face_image_callback(self, face_msg: FaceEmbedding, image_msg: Image) -> None:
        """比對人臉向量訊息進行身份驗證"""
        face_embedding = np.array(face_msg.embedding, dtype=np.float32)
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        with self.face_lock:
            if self.is_registering_face and self.current_registration:
                user_uuid = self.current_registration.user_uuid
                self.user_manager.add_face_sample(user_uuid, face_embedding, cv_image)
                self.current_registration.collected_samples += 1

                if self.current_registration.collected_samples >= self.current_registration.target_samples:
                    if self.face_reg_timer:
                        self.face_reg_timer.cancel()

                    self.is_registering_face = False
                    self.get_logger().info(f"用戶 {self.current_registration.user_name} 人臉註冊成功")
                    self.current_registration = None
                    return

            user_uuid = self._process_face_recognition(face_embedding)
            if user_uuid:
                user_info = self.user_manager.get_user_info(user_uuid)
                if user_info:
                    user_name = user_info["user_name"]
                    user_type = user_info["user_type"]
                    description = user_info["description"]
                    self.get_logger().info(
                        f"身份驗證成功: {user_name} (UUID: {user_uuid}, 類型: {user_type}, 描述: {description})"
                    )

    def _process_face_recognition(self, embedding: np.ndarray) -> Optional[str]:
        """處理人臉識別邏輯"""

        all_embeddings = self.user_manager.get_all_embeddings()

        max_similarity = 0.0
        best_match_uuid = None

        for user_uuid, embeddings in all_embeddings.items():
            mean_embedding = cast(np.ndarray, np.mean(embeddings, axis=0, dtype=np.float32))
            similarity = compute_similarity(embedding, mean_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_match_uuid = user_uuid

        return best_match_uuid


def main(args=None):
    """使用者身份驗證節點進入點"""
    rclpy.init(args=args)
    node = UserAuthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
