import logging
import os
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("video_service.log")],
)
logger = logging.getLogger("VideoProcessingService")


class VideoProcessingService:
    """Dịch vụ xử lý video trung tâm kết nối với RabbitMQ và Cloudflare R2"""

    def __init__(
        self,
        rabbitmq_host="localhost",
        rabbitmq_port=5672,
        rabbitmq_username="guest",
        rabbitmq_password="guest",
        exchange="video-processing-exchange",
        input_queue="video-processing-queue",
        output_queue="video-result-queue",
        input_routing_key="video-processing",
        output_routing_key="video-result",
        r2_access_key=None,
        r2_secret_key=None,
        r2_endpoint=None,
        r2_region=None,
        r2_bucket_name=None,
        player_model_path="data/player.pt",
        ball_model_path="data/ball.pt",
        device="cuda",
    ):
        """Khởi tạo dịch vụ xử lý video

        Args:
            rabbitmq_host: Host của RabbitMQ server
            rabbitmq_port: Port của RabbitMQ server
            rabbitmq_username: Tên đăng nhập RabbitMQ
            rabbitmq_password: Mật khẩu RabbitMQ
            exchange: Exchange để gửi và nhận tin nhắn
            input_queue: Hàng đợi để nhận yêu cầu xử lý
            output_queue: Hàng đợi để gửi kết quả
            input_routing_key: Routing key cho yêu cầu xử lý
            output_routing_key: Routing key cho kết quả
            r2_access_key: R2 access key
            r2_secret_key: R2 secret key
            r2_endpoint: R2 endpoint URL
            r2_region: R2 region
            r2_bucket_name: Tên bucket R2
            player_model_path: Đường dẫn đến mô hình YOLO phát hiện cầu thủ
            ball_model_path: Đường dẫn đến mô hình YOLO phát hiện bóng
            device: Thiết bị xử lý (cuda hoặc cpu)
        """
        # Import các module mà chúng ta cần
        from utils.r2_client import R2Client
        from utils.rabbitmq_client import RabbitMQClient
        from utils.team_classifier import TeamClassifier
        from utils.video_processor import VideoProcessor

        # Khởi tạo client RabbitMQ
        self.rabbitmq_client = RabbitMQClient(
            host=rabbitmq_host,
            port=rabbitmq_port,
            username=rabbitmq_username,
            password=rabbitmq_password,
            exchange=exchange,
            input_queue=input_queue,
            output_queue=output_queue,
            input_routing_key=input_routing_key,
            output_routing_key=output_routing_key,
        )

        # Khởi tạo client R2
        self.r2_client = R2Client(
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            endpoint_url=r2_endpoint,
            region=r2_region,
            bucket_name=r2_bucket_name,
        )

        # Tạo thư mục tạm để lưu video
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Thư mục tạm: {self.temp_dir}")

        # Thiết bị xử lý
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA không khả dụng, chuyển sang CPU")
            self.device = "cpu"

        # Tải mô hình YOLO
        logger.info(
            f"Đang tải mô hình YOLO từ: {player_model_path} và {ball_model_path}"
        )
        try:
            self.player_detection_model = YOLO(player_model_path).to(self.device)
            self.ball_detection_model = YOLO(ball_model_path).to(self.device)

            # Fuse models nếu dùng GPU
            if self.device == "cuda":
                self.player_detection_model.model.fuse()
                self.ball_detection_model.model.fuse()

            logger.info("Tải mô hình YOLO thành công")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình YOLO: {e}")
            raise

        # Khởi tạo bộ phân loại đội
        self.team_classifier = TeamClassifier(device=self.device)

        # Khởi tạo bộ xử lý video
        self.video_processor = VideoProcessor()

        # Lưu thời gian cập nhật tiến độ cuối cùng cho mỗi video
        self.last_progress_update = {}

        # Khoảng thời gian giữa các lần cập nhật tiến độ (giây)
        self.progress_update_interval = 10

        logger.info("Khởi tạo dịch vụ xử lý video thành công")

    def start(self):
        """Khởi động dịch vụ và bắt đầu tiêu thụ tin nhắn từ RabbitMQ"""
        logger.info("Bắt đầu dịch vụ xử lý video...")

        # Kết nối đến RabbitMQ
        if not self.rabbitmq_client.connect():
            logger.error("Không thể kết nối đến RabbitMQ")
            return

        # Bắt đầu tiêu thụ tin nhắn
        self.rabbitmq_client.start_consuming(self.process_message)

        # Giữ chương trình chạy
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Nhận lệnh dừng dịch vụ...")
        finally:
            self.stop()

    def stop(self):
        """Dừng dịch vụ"""
        logger.info("Đang dừng dịch vụ xử lý video...")

        # Dừng tiêu thụ tin nhắn
        self.rabbitmq_client.stop_consuming()

        # Đóng kết nối
        self.rabbitmq_client.close()

        logger.info("Dịch vụ xử lý video đã dừng")

    def process_message(self, message):
        """Xử lý tin nhắn nhận được từ RabbitMQ

        Args:
            message: Tin nhắn xử lý video
        """

        if not message.videoId or not message.cloudStorageKey or not message.outputPath:
            logger.error(
                f"Thiếu thông tin trong tin nhắn: videoId={message.videoId}, cloudStorageKey={message.cloudStorageKey}, outputPath={message.outputPath}"
            )
            self._send_error_result(message, "Thiếu thông tin trong tin nhắn")
            return

        try:
            logger.info(f"Bắt đầu xử lý video ID: {message.videoId}")

            # Cập nhật trạng thái thành PROCESSING
            message.status = "PROCESSING"
            message.progress = 0
            message.message = "Bắt đầu xử lý video"
            message.timestamp = datetime.now().isoformat()
            self.rabbitmq_client.send_result(message)

            # Tạo đường dẫn tải video xuống
            video_extension = Path(message.cloudStorageKey).suffix
            if not video_extension:
                video_extension = ".mp4"  # Mặc định nếu không có đuôi file
            download_path = self.temp_dir / f"{message.videoId}{video_extension}"

            # Tải video từ R2
            logger.info(f"Đang tải video từ R2: {message.cloudStorageKey}")
            if not self.r2_client.download_file(
                message.cloudStorageKey, str(download_path)
            ):
                raise Exception(f"Không thể tải video từ R2: {message.cloudStorageKey}")

            # Cập nhật tiến độ
            message.progress = 10
            message.message = "Đã tải video, bắt đầu xử lý"
            self.rabbitmq_client.send_result(message)

            # Thiết lập thời gian cập nhật tiến độ cuối cùng
            self.last_progress_update[message.videoId] = time.time()

            # Tạo đường dẫn để lưu video đã xử lý
            output_filename = f"{Path(download_path).stem}_processed{video_extension}"
            processed_path = self.temp_dir / output_filename

            # Xử lý video
            logger.info(f"Bắt đầu xử lý video: {download_path}")
            self.video_processor.process_soccer_video(
                source_video_path=str(download_path),
                target_video_path=str(processed_path),
                player_detection_model=self.player_detection_model,
                ball_detection_model=self.ball_detection_model,
                team_classifier=self.team_classifier,
                device=self.device,
                progress_callback=lambda progress: self._update_progress(
                    message, progress
                ),
            )

            # Kiểm tra file đầu ra
            if not processed_path.exists():
                raise Exception(f"Không tìm thấy file đã xử lý: {processed_path}")

            logger.info(f"Video đã xử lý (có âm thanh) được lưu tại: {processed_path}")

            # Cập nhật tiến độ
            message.progress = 80
            message.message = "Đã xử lý video, đang nén video để giảm dung lượng"
            self.rabbitmq_client.send_result(message)

            # Nén video để giảm dung lượng nhưng giữ chất lượng
            compressed_path = (
                self.temp_dir
                / f"{Path(download_path).stem}_compressed{video_extension}"
            )
            logger.info(
                f"Đang nén video để giảm dung lượng: {processed_path} -> {compressed_path}"
            )
            compression_success = self.video_processor.compress_video(
                str(processed_path),
                str(compressed_path),
                crf=23,  # 18-28: thấp hơn = chất lượng cao hơn
                preset="medium",  # Cân bằng tốc độ và chất lượng
            )

            # Sử dụng file đã nén nếu thành công, nếu không thì dùng file gốc
            upload_path = (
                compressed_path
                if compression_success and compressed_path.exists()
                else processed_path
            )

            # Cập nhật tiến độ
            message.progress = 85
            message.message = "Đã nén video, đang tải lên R2"
            self.rabbitmq_client.send_result(message)

            # Tải video đã xử lý lên R2
            logger.info(f"Đang tải video đã xử lý lên R2: {message.outputPath}")
            video_metadata = {
                "original-name": Path(message.cloudStorageKey).name,
                "user-id": message.userId,
                "processed": "true",
            }

            if not self.r2_client.upload_file(
                str(upload_path),
                message.outputPath,
                metadata=video_metadata,
            ):
                raise Exception(
                    f"Không thể tải video đã xử lý lên R2: {message.outputPath}"
                )

            # Xong! Gửi kết quả
            message.status = "COMPLETED"
            message.progress = 100
            message.message = "Xử lý video hoàn tất (file kết quả lưu ở thư mục tạm)"
            message.timestamp = datetime.now().isoformat()
            self.rabbitmq_client.send_result(message)

            # Xóa các file tạm
            if download_path.exists():
                os.remove(download_path)
            if processed_path.exists():
                os.remove(processed_path)
            if compression_success and compressed_path.exists():
                os.remove(compressed_path)

            # Xóa thông tin cập nhật tiến độ cho video này
            if message.videoId in self.last_progress_update:
                del self.last_progress_update[message.videoId]

            logger.info(f"Hoàn thành xử lý video ID: {message.videoId}")

        except Exception as e:
            logger.error(f"Lỗi khi xử lý video ID {message.videoId}: {e}")
            self._send_error_result(message, str(e))

            # Kiểm tra có nên thử lại không
            if message.retryCount < message.maxRetries:
                message.retryCount += 1
                message.status = "PENDING"
                message.progress = 0
                message.message = f"Thử lại {message.retryCount}/{message.maxRetries}"
                message.timestamp = datetime.now().isoformat()

                logger.info(
                    f"Thử lại xử lý video ID {message.videoId} lần {message.retryCount}/{message.maxRetries}"
                )
                # Gửi lại tin nhắn để thử lại sau
                self.rabbitmq_client.send_retry(message)

    def _send_error_result(self, message, error_message):
        """Gửi thông báo lỗi"""
        message.status = "ERROR"
        message.message = error_message
        message.timestamp = datetime.now().isoformat()
        self.rabbitmq_client.send_result(message)

    def _update_progress(self, message, progress):
        """Cập nhật tiến độ xử lý"""
        # Đảm bảo tiến độ từ 10-80% (đầu và cuối được xử lý bởi các giai đoạn khác)
        adjusted_progress = int(10 + progress * 0.7)  # Chuyển đổi 0-100 thành 10-80

        if adjusted_progress > message.progress:
            current_time = time.time()
            video_id = message.videoId

            # Kiểm tra thời gian từ lần cập nhật cuối
            if video_id not in self.last_progress_update:
                # Đây là lần cập nhật đầu tiên
                should_update = True
            else:
                # Kiểm tra đã đủ thời gian chưa
                elapsed = current_time - self.last_progress_update[video_id]
                should_update = elapsed >= self.progress_update_interval

            # Cập nhật nếu đủ điều kiện
            if should_update:
                message.progress = adjusted_progress
                message.message = f"Đang xử lý video: {adjusted_progress}%"
                self.rabbitmq_client.send_result(message)
                self.last_progress_update[video_id] = current_time
                logger.debug(
                    f"Đã gửi cập nhật tiến độ: {adjusted_progress}% cho video ID: {video_id}"
                )
