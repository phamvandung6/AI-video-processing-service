#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path

# Đảm bảo các thư mục utils và src có thể import được
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
logger = logging.getLogger("RunService")


def main():
    """Điểm khởi chạy chính của dịch vụ xử lý video"""

    # Tải biến môi trường (nếu có file .env)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.info("Đã tải biến môi trường từ file .env")
    except ImportError:
        logger.warning("Không tìm thấy thư viện python-dotenv, bỏ qua việc tải .env")

    # Xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Dịch vụ xử lý video bóng đá kết nối RabbitMQ và Cloudflare R2"
    )
    parser.add_argument(
        "--rabbitmq-host",
        default=os.getenv("RABBITMQ_HOST", "localhost"),
        help="RabbitMQ host",
    )
    parser.add_argument(
        "--rabbitmq-port",
        type=int,
        default=int(os.getenv("RABBITMQ_PORT", "5672")),
        help="RabbitMQ port",
    )
    parser.add_argument(
        "--rabbitmq-username",
        default=os.getenv("RABBITMQ_USERNAME", "guest"),
        help="RabbitMQ username",
    )
    parser.add_argument(
        "--rabbitmq-password",
        default=os.getenv("RABBITMQ_PASSWORD", "guest"),
        help="RabbitMQ password",
    )
    parser.add_argument(
        "--exchange",
        default=os.getenv("RABBITMQ_EXCHANGE", "video-processing-exchange"),
        help="RabbitMQ exchange",
    )
    parser.add_argument(
        "--input-queue",
        default=os.getenv("RABBITMQ_INPUT_QUEUE", "video-processing-queue"),
        help="RabbitMQ input queue",
    )
    parser.add_argument(
        "--output-queue",
        default=os.getenv("RABBITMQ_OUTPUT_QUEUE", "video-result-queue"),
        help="RabbitMQ output queue",
    )
    parser.add_argument(
        "--input-routing-key",
        default=os.getenv("RABBITMQ_INPUT_ROUTING_KEY", "video-processing"),
        help="RabbitMQ input routing key",
    )
    parser.add_argument(
        "--output-routing-key",
        default=os.getenv("RABBITMQ_OUTPUT_ROUTING_KEY", "video-result"),
        help="RabbitMQ output routing key",
    )
    parser.add_argument(
        "--r2-access-key", default=os.getenv("R2_ACCESS_KEY"), help="R2 access key"
    )
    parser.add_argument(
        "--r2-secret-key", default=os.getenv("R2_SECRET_KEY"), help="R2 secret key"
    )
    parser.add_argument(
        "--r2-endpoint", default=os.getenv("R2_ENDPOINT"), help="R2 endpoint URL"
    )
    parser.add_argument(
        "--r2-region", default=os.getenv("R2_REGION", "auto"), help="R2 region"
    )
    parser.add_argument(
        "--r2-bucket-name",
        default=os.getenv("R2_BUCKET_NAME", "soccer-videos"),
        help="R2 bucket name",
    )
    parser.add_argument(
        "--player-model-path",
        default=os.getenv("PLAYER_MODEL_PATH", "data/player.pt"),
        help="YOLO player model path",
    )
    parser.add_argument(
        "--ball-model-path",
        default=os.getenv("BALL_MODEL_PATH", "data/ball.pt"),
        help="YOLO ball model path",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "cuda"),
        help="Device to run models (cuda/cpu)",
    )

    args = parser.parse_args()

    # Kiểm tra các tham số bắt buộc
    required_params = [
        "r2_access_key",
        "r2_secret_key",
        "r2_endpoint",
        "r2_bucket_name",
    ]
    missing_params = [
        param for param in required_params if not getattr(args, param.replace("-", "_"))
    ]

    if missing_params:
        for param in missing_params:
            logger.error(f"Thiếu tham số bắt buộc: {param}")
        return 1

    # Kiểm tra tồn tại của mô hình YOLO
    player_model_path = Path(args.player_model_path)
    ball_model_path = Path(args.ball_model_path)

    if not player_model_path.exists():
        logger.error(f"Không tìm thấy mô hình player: {player_model_path}")
        return 1
    if not ball_model_path.exists():
        logger.error(f"Không tìm thấy mô hình ball: {ball_model_path}")
        return 1

    # Log cấu hình
    logger.info(f"Sử dụng RabbitMQ: {args.rabbitmq_host}:{args.rabbitmq_port}")
    logger.info(f"Sử dụng R2 Bucket: {args.r2_bucket_name}")
    logger.info(f"Sử dụng mô hình Player: {args.player_model_path}")
    logger.info(f"Sử dụng mô hình Ball: {args.ball_model_path}")
    logger.info(f"Sử dụng thiết bị: {args.device}")

    # Khởi tạo dịch vụ
    try:
        from src.video_processing_service import VideoProcessingService

        service = VideoProcessingService(
            rabbitmq_host=args.rabbitmq_host,
            rabbitmq_port=args.rabbitmq_port,
            rabbitmq_username=args.rabbitmq_username,
            rabbitmq_password=args.rabbitmq_password,
            exchange=args.exchange,
            input_queue=args.input_queue,
            output_queue=args.output_queue,
            input_routing_key=args.input_routing_key,
            output_routing_key=args.output_routing_key,
            r2_access_key=args.r2_access_key,
            r2_secret_key=args.r2_secret_key,
            r2_endpoint=args.r2_endpoint,
            r2_region=args.r2_region,
            r2_bucket_name=args.r2_bucket_name,
            player_model_path=args.player_model_path,
            ball_model_path=args.ball_model_path,
            device=args.device,
        )

        # Khởi động dịch vụ
        logger.info("Bắt đầu chạy dịch vụ xử lý video...")
        service.start()

        return 0
    except Exception as e:
        logger.error(f"Lỗi khi khởi động dịch vụ: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
