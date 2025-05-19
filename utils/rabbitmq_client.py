import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict

import pika

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RabbitMQClient")


class VideoProcessingMessage:
    """Cấu trúc tin nhắn xử lý video"""

    def __init__(self):
        self.videoId = None  # UUID
        self.userId = None  # UUID
        self.videoPath = None  # String - đường dẫn đến video gốc
        self.cloudStorageKey = None  # String - key lưu trữ trên R2
        self.outputPath = None  # String - đường dẫn lưu kết quả
        self.status = (
            None  # String - trạng thái xử lý (PENDING, PROCESSING, COMPLETED, ERROR)
        )
        self.progress = None  # Integer - tiến độ xử lý (0-100)
        self.message = None  # String - thông điệp xử lý
        self.timestamp = None  # String - thời gian (ISO format)
        self.highPriority = False  # Boolean - ưu tiên cao hay không
        self.maxRetries = 3  # Integer - số lần thử lại tối đa
        self.retryCount = 0  # Integer - số lần đã thử lại

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoProcessingMessage":
        """Tạo đối tượng từ dictionary"""
        message = cls()
        message.videoId = data.get("videoId")
        message.userId = data.get("userId")
        message.videoPath = data.get("videoPath")
        message.cloudStorageKey = data.get("cloudStorageKey")
        message.outputPath = data.get("outputPath")
        message.status = data.get("status", "PENDING")
        message.progress = data.get("progress", 0)
        message.message = data.get("message", "")
        message.timestamp = data.get("timestamp", datetime.now().isoformat())
        message.highPriority = data.get("highPriority", False)
        message.maxRetries = data.get("maxRetries", 3)
        message.retryCount = data.get("retryCount", 0)
        return message

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đối tượng thành dictionary"""
        return {
            "videoId": self.videoId,
            "userId": self.userId,
            "videoPath": self.videoPath,
            "cloudStorageKey": self.cloudStorageKey,
            "outputPath": self.outputPath,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "timestamp": self.timestamp,
            "highPriority": self.highPriority,
            "maxRetries": self.maxRetries,
            "retryCount": self.retryCount,
        }


class RabbitMQClient:
    """Client để tương tác với RabbitMQ"""

    def __init__(
        self,
        host="localhost",
        port=5672,
        username="guest",
        password="guest",
        exchange="video-processing-exchange",
        input_queue="video-processing-queue",
        output_queue="video-result-queue",
        input_routing_key="video-processing",
        output_routing_key="video-result",
    ):
        """Khởi tạo RabbitMQ client

        Args:
            host: Host của RabbitMQ server
            port: Port của RabbitMQ server
            username: Tên đăng nhập
            password: Mật khẩu
            exchange: Exchange để gửi và nhận tin nhắn
            input_queue: Hàng đợi để nhận yêu cầu xử lý
            output_queue: Hàng đợi để gửi kết quả
            input_routing_key: Routing key cho yêu cầu xử lý
            output_routing_key: Routing key cho kết quả
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange = exchange
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.input_routing_key = input_routing_key
        self.output_routing_key = output_routing_key

        self.connection = None
        self.channel = None
        self.consumer_thread = None
        self.is_consuming = False

        logger.info(f"RabbitMQ Client đã được khởi tạo: {host}:{port}")

    def connect(self) -> bool:
        """Kết nối đến RabbitMQ server

        Returns:
            bool: True nếu kết nối thành công, False nếu thất bại
        """
        try:
            # Tạo credentials
            credentials = pika.PlainCredentials(self.username, self.password)

            # Tạo connection parameters
            parameters = pika.ConnectionParameters(
                host=self.host, port=self.port, credentials=credentials, heartbeat=600
            )

            # Kết nối đến RabbitMQ
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Khai báo exchange - nếu chưa tồn tại
            self.channel.exchange_declare(
                exchange=self.exchange, exchange_type="direct", durable=True
            )

            # Khai báo retry exchange
            retry_exchange = f"{self.exchange}.retry"
            self.channel.exchange_declare(
                exchange=retry_exchange, exchange_type="direct", durable=True
            )

            # Đặt tên đầy đủ cho dead letter exchange
            dead_letter_exchange = self.exchange

            # Khai báo queue đầu vào với dead letter exchange
            queue_args_input = {
                "x-dead-letter-exchange": dead_letter_exchange,
                "x-dead-letter-routing-key": self.input_routing_key,
            }

            # Khai báo queue đầu vào
            try:
                self.channel.queue_declare(
                    queue=self.input_queue, durable=True, arguments=queue_args_input
                )
            except pika.exceptions.ChannelClosedByBroker:
                # Nếu queue đã tồn tại với thuộc tính khác, chỉ kết nối lại
                logger.warning(
                    f"Queue {self.input_queue} đã tồn tại với thuộc tính khác, thử kết nối lại..."
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()

            # Khai báo queue đầu ra
            try:
                self.channel.queue_declare(queue=self.output_queue, durable=True)
            except pika.exceptions.ChannelClosedByBroker:
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()

            # Khai báo retry queue
            retry_queue = f"{self.input_queue}.retry"
            retry_args = {
                "x-dead-letter-exchange": self.exchange,
                "x-dead-letter-routing-key": self.input_routing_key,
                "x-message-ttl": 30000,  # 30 seconds delay
            }
            try:
                self.channel.queue_declare(
                    queue=retry_queue, durable=True, arguments=retry_args
                )
            except pika.exceptions.ChannelClosedByBroker:
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()

            # Binding
            self.channel.queue_bind(
                exchange=self.exchange,
                queue=self.input_queue,
                routing_key=self.input_routing_key,
            )

            self.channel.queue_bind(
                exchange=self.exchange,
                queue=self.output_queue,
                routing_key=self.output_routing_key,
            )

            self.channel.queue_bind(
                exchange=retry_exchange,
                queue=retry_queue,
                routing_key=self.input_routing_key,
            )

            logger.info(f"Kết nối thành công đến RabbitMQ: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi kết nối đến RabbitMQ: {e}")
            return False

    def close(self):
        """Đóng kết nối đến RabbitMQ"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Đã đóng kết nối RabbitMQ")
            self.connection = None
            self.channel = None

    def start_consuming(
        self, callback: Callable[[VideoProcessingMessage], None]
    ) -> bool:
        """Bắt đầu tiêu thụ tin nhắn từ hàng đợi

        Args:
            callback: Hàm xử lý tin nhắn nhận được

        Returns:
            bool: True nếu bắt đầu tiêu thụ thành công, False nếu thất bại
        """
        if not self.connection or not self.channel:
            if not self.connect():
                return False

        def on_message(ch, method, properties, body):
            try:
                # Phân tích tin nhắn nhận được
                message_dict = json.loads(body)
                message = VideoProcessingMessage.from_dict(message_dict)

                # Gọi callback
                callback(message)

                # Xác nhận đã xử lý tin nhắn
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý tin nhắn: {e}")
                # Từ chối và đưa lại vào hàng đợi nếu cần
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        # Cấu hình prefetch count
        self.channel.basic_qos(prefetch_count=1)

        # Bắt đầu tiêu thụ
        self.channel.basic_consume(
            queue=self.input_queue, on_message_callback=on_message
        )

        # Đánh dấu đang tiêu thụ
        self.is_consuming = True

        # Tạo luồng tiêu thụ
        def consume_thread():
            logger.info(f"Bắt đầu tiêu thụ tin nhắn từ hàng đợi: {self.input_queue}")
            try:
                self.channel.start_consuming()
            except Exception as e:
                logger.error(f"Lỗi trong quá trình tiêu thụ: {e}")
            finally:
                self.is_consuming = False
                logger.info("Đã dừng tiêu thụ tin nhắn")

        self.consumer_thread = threading.Thread(target=consume_thread)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()

        return True

    def stop_consuming(self):
        """Dừng tiêu thụ tin nhắn"""
        if self.is_consuming and self.channel:
            self.channel.stop_consuming()
            if self.consumer_thread:
                self.consumer_thread.join(timeout=5.0)
            self.is_consuming = False
            logger.info("Đã dừng tiêu thụ tin nhắn")

    def send_result(self, message: VideoProcessingMessage) -> bool:
        """Gửi kết quả xử lý vào hàng đợi kết quả

        Args:
            message: Tin nhắn kết quả cần gửi

        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        try:
            if not self.connection or not self.channel:
                if not self.connect():
                    return False

            # Cập nhật timestamp
            if not message.timestamp:
                message.timestamp = datetime.now().isoformat()

            # Chuyển đối tượng thành JSON
            message_body = json.dumps(message.to_dict())

            # Gửi tin nhắn
            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=self.output_routing_key,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Đảm bảo tin nhắn bền vững
                    content_type="application/json",
                    priority=5 if message.highPriority else 1,
                ),
            )
            logger.info(f"Đã gửi kết quả cho video ID: {message.videoId}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi gửi kết quả: {e}")
            return False

    def send_retry(
        self, message: VideoProcessingMessage, delay_seconds: int = 30
    ) -> bool:
        """Gửi lại tin nhắn vào hàng đợi để thử lại sau một khoảng thời gian

        Args:
            message: Tin nhắn cần gửi lại
            delay_seconds: Số giây trì hoãn trước khi xử lý lại

        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        try:
            if not self.connection or not self.channel:
                if not self.connect():
                    return False

            # Chuyển đối tượng thành JSON
            message_body = json.dumps(message.to_dict())

            # Tạo exchange trì hoãn nếu chưa tồn tại
            retry_exchange = f"{self.exchange}.retry"

            # Gửi tin nhắn đến exchange trì hoãn
            self.channel.basic_publish(
                exchange=retry_exchange,
                routing_key=self.input_routing_key,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                    priority=5 if message.highPriority else 1,
                ),
            )
            logger.info(
                f"Đã đặt lịch thử lại cho video ID: {message.videoId} sau {delay_seconds}s"
            )
            return True
        except Exception as e:
            logger.error(f"Lỗi khi gửi tin nhắn thử lại: {e}")
            return False
