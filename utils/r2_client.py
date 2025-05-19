import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("R2Client")


class R2Client:
    """Client để tương tác với Cloudflare R2 Storage (tương thích S3)"""

    def __init__(
        self,
        access_key=None,
        secret_key=None,
        endpoint_url=None,
        region=None,
        bucket_name=None,
    ):
        """Khởi tạo R2 client

        Args:
            access_key: R2 access key
            secret_key: R2 secret key
            endpoint_url: R2 endpoint URL
            region: R2 region
            bucket_name: Tên bucket
        """
        # Lấy giá trị từ biến môi trường nếu không được cung cấp
        self.access_key = access_key or os.getenv("R2_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("R2_SECRET_KEY")
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT")
        self.region = region or os.getenv("R2_REGION", "auto")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "soccer-videos")

        # Kiểm tra các tham số bắt buộc
        if not all(
            [self.access_key, self.secret_key, self.endpoint_url, self.bucket_name]
        ):
            missing = []
            if not self.access_key:
                missing.append("R2_ACCESS_KEY")
            if not self.secret_key:
                missing.append("R2_SECRET_KEY")
            if not self.endpoint_url:
                missing.append("R2_ENDPOINT")
            if not self.bucket_name:
                missing.append("R2_BUCKET_NAME")
            raise ValueError(f"Thiếu thông tin cấu hình R2: {', '.join(missing)}")

        # Khởi tạo client S3
        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region,
        )

        # Khởi tạo resource S3
        self.resource = boto3.resource(
            "s3",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region,
        )

        # Lấy tham chiếu đến bucket
        self.bucket = self.resource.Bucket(self.bucket_name)

        logger.info(f"R2 Client đã được khởi tạo cho bucket {self.bucket_name}")

    def download_file(self, object_key, local_path):
        """Tải file từ R2 về local

        Args:
            object_key: Đường dẫn của đối tượng trên R2
            local_path: Đường dẫn cục bộ để lưu file

        Returns:
            bool: True nếu tải thành công, False nếu thất bại
        """
        try:
            # Đảm bảo thư mục cha tồn tại
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Đang tải file từ R2: {object_key} -> {local_path}")
            self.client.download_file(self.bucket_name, object_key, local_path)
            return True
        except ClientError as e:
            logger.error(f"Lỗi khi tải file từ R2: {e}")
            return False

    def upload_file(self, local_path, object_key, metadata=None, content_type=None):
        """Tải file lên R2

        Args:
            local_path: Đường dẫn cục bộ của file cần tải lên
            object_key: Đường dẫn lưu trữ trên R2
            metadata: Metadata cho file
            content_type: Kiểu nội dung của file

        Returns:
            bool: True nếu tải lên thành công, False nếu thất bại
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            if content_type:
                extra_args["ContentType"] = content_type
            else:
                # Đoán content type dựa trên đuôi file
                import mimetypes

                file_ext = Path(local_path).suffix.lower()
                guessed_type = mimetypes.guess_type(local_path)[0]
                if guessed_type:
                    extra_args["ContentType"] = guessed_type
                elif file_ext in [".mp4", ".mov", ".avi"]:
                    extra_args["ContentType"] = f"video/{file_ext[1:]}"

            logger.info(f"Đang tải file lên R2: {local_path} -> {object_key}")
            self.client.upload_file(
                local_path, self.bucket_name, object_key, ExtraArgs=extra_args
            )
            return True
        except ClientError as e:
            logger.error(f"Lỗi khi tải file lên R2: {e}")
            return False

    def check_file_exists(self, object_key):
        """Kiểm tra xem file có tồn tại trên R2 không

        Args:
            object_key: Đường dẫn của đối tượng trên R2

        Returns:
            bool: True nếu file tồn tại, False nếu không
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError:
            return False

    def list_files(self, prefix=None):
        """Liệt kê các file trong bucket

        Args:
            prefix: Tiền tố để lọc danh sách file

        Returns:
            list: Danh sách các file
        """
        try:
            if prefix:
                objects = self.bucket.objects.filter(Prefix=prefix)
            else:
                objects = self.bucket.objects.all()
            return [obj.key for obj in objects]
        except ClientError as e:
            logger.error(f"Lỗi khi liệt kê file từ R2: {e}")
            return []

    def delete_file(self, object_key):
        """Xóa file khỏi R2

        Args:
            object_key: Đường dẫn của đối tượng trên R2

        Returns:
            bool: True nếu xóa thành công, False nếu thất bại
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            logger.error(f"Lỗi khi xóa file từ R2: {e}")
            return False
