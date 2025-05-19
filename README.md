# Dịch Vụ Phân Tích Video Bóng Đá

Dự án phân tích video bóng đá sử dụng AI, thiết kế theo kiến trúc microservice để xử lý video, phát hiện cầu thủ, bóng, và phân loại các đội bóng.

## Tính Năng

- Phát hiện đối tượng: cầu thủ, thủ môn, bóng và trọng tài
- Phân loại đội dựa trên mô hình SiglipVision, UMAP và KMeans
- Theo dõi cầu thủ và quả bóng qua các khung hình
- Hỗ trợ xử lý bất đồng bộ với RabbitMQ
- Tích hợp lưu trữ đám mây với Cloudflare R2
- Hỗ trợ thử lại tự động khi xử lý thất bại
- Cơ chế báo cáo tiến độ theo thời gian thực

## Yêu Cầu Hệ Thống

- Python 3.8+
- GPU tương thích CUDA (khuyến nghị)
- RabbitMQ Server
- Tài khoản Cloudflare R2

## Cấu Trúc Dự Án

```
.
├── src/                     # Mã nguồn dịch vụ chính
├── utils/                   # Các tiện ích hỗ trợ
├── data/                    # Chứa các mô hình đã huấn luyện
├── output_videos/           # Thư mục đầu ra video
├── run_service.py           # Script khởi chạy
├── requirements.txt         # Các phụ thuộc
└── README.md                # File này
```

## Cài Đặt

1. Tạo môi trường ảo:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # hoặc
   .venv\Scripts\activate  # Windows
   ```

2. Cài đặt phụ thuộc:

   ```bash
   pip install -r requirements.txt
   ```

3. Cấu hình môi trường:
   ```bash
   cp env-example .env
   # Chỉnh sửa .env với thông tin của bạn
   ```

## Sử Dụng

### Khởi Động Dịch Vụ

```bash
python run_service.py
```

Hoặc với các tham số cụ thể:

```bash
python run_service.py --rabbitmq-host localhost --r2-bucket-name my-bucket
```

### Gửi Yêu Cầu Xử Lý

Để xử lý video, gửi tin nhắn JSON đến RabbitMQ với định dạng sau:

```json
{
  "videoId": "unique-video-id",
  "userId": "user-id",
  "cloudStorageKey": "path/to/video/in/r2",
  "outputPath": "path/to/store/processed/video",
  "status": "PENDING",
  "highPriority": false,
  "maxRetries": 3
}
```

### Nhận Kết Quả

Kết quả xử lý sẽ được gửi đến hàng đợi đầu ra với định dạng tương tự, bao gồm thêm thông tin về tiến độ, trạng thái và thông báo.

## Triển Khai với Docker

1. Xây dựng Docker image:

   ```bash
   docker build -t soccer-video-analysis-service .
   ```

2. Chạy container:
   ```bash
   docker run -d \
     --name soccer-video-service \
     -e RABBITMQ_HOST=your-rabbitmq-host \
     -e R2_ACCESS_KEY=your-access-key \
     -e R2_SECRET_KEY=your-secret-key \
     -e R2_ENDPOINT=your-endpoint \
     -e R2_BUCKET_NAME=your-bucket \
     --gpus all \
     soccer-video-analysis-service
   ```

## Giải Thích Microservice

### Các Thành Phần Chính

1. **Video Processing Service**: Xử lý chính cho video
2. **RabbitMQ Client**: Quản lý hàng đợi tin nhắn
3. **R2 Client**: Quản lý lưu trữ đám mây
4. **Team Classifier**: AI phân loại đội từ hình ảnh

### Luồng Tin Nhắn

1. Frontend gửi yêu cầu xử lý đến hàng đợi đầu vào
2. Dịch vụ nhận yêu cầu, tải video từ R2
3. Xử lý video với mô hình AI
4. Tải video đã xử lý lên R2
5. Gửi thông báo kết quả đến hàng đợi đầu ra
