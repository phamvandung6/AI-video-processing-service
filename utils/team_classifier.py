import logging
from typing import Generator, Iterable, List, TypeVar

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TeamClassifier")

V = TypeVar("V")
SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Tạo batches từ một chuỗi với kích thước batch được chỉ định.

    Args:
        sequence: Chuỗi đầu vào cần chia batch
        batch_size: Kích thước mỗi batch

    Yields:
        Generator[List[V], None, None]: Generator tạo ra các batches
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    Bộ phân loại sử dụng SiglipVisionModel cho trích xuất đặc trưng,
    UMAP để giảm chiều dữ liệu, và KMeans để phân cụm.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        """
        Khởi tạo TeamClassifier với thiết bị và kích thước batch.

        Args:
            device: Thiết bị để chạy mô hình ('cpu' hoặc 'cuda')
            batch_size: Kích thước batch khi xử lý hình ảnh
        """
        self.device = device
        self.batch_size = batch_size
        logger.info(f"Tải mô hình SiglipVision từ {SIGLIP_MODEL_PATH}")
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(
            device
        )
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        logger.info(
            f"Đã khởi tạo TeamClassifier với device={device}, batch_size={batch_size}"
        )

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Trích xuất đặc trưng từ danh sách crops hình ảnh dùng SiglipVisionModel.

        Args:
            crops: Danh sách các crop hình ảnh

        Returns:
            np.ndarray: Đặc trưng được trích xuất
        """
        import supervision as sv

        # Chuyển đổi crops từ cv2 (BGR) sang PIL (RGB)
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc="Extracting features"):
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                outputs = self.features_model(**inputs)
                # Mean pooling trên chiều dài chuỗi để tạo embedding cố định
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        if not data:  # Kiểm tra nếu không có dữ liệu
            logger.warning("Không có đặc trưng nào được trích xuất!")
            return np.array([])

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Huấn luyện mô hình trên danh sách crops hình ảnh.

        Args:
            crops: Danh sách các crop hình ảnh
        """
        if not crops:
            logger.error("Không có crops nào để huấn luyện!")
            raise ValueError("Danh sách crops trống")

        logger.info(f"Bắt đầu trích xuất đặc trưng từ {len(crops)} crops")
        data = self.extract_features(crops)

        if data.size == 0:
            logger.error("Không trích xuất được đặc trưng!")
            raise ValueError("Trích xuất đặc trưng thất bại")

        logger.info("Giảm chiều dữ liệu bằng UMAP")
        projections = self.reducer.fit_transform(data)

        logger.info("Phân cụm dữ liệu bằng KMeans")
        self.cluster_model.fit(projections)
        logger.info("Đã hoàn thành huấn luyện mô hình phân loại đội")

        # Phân tích kết quả phân cụm
        team_0_count = np.sum(self.cluster_model.labels_ == 0)
        team_1_count = np.sum(self.cluster_model.labels_ == 1)
        logger.info(
            f"Phân phối phân cụm: Nhóm 0: {team_0_count}, Nhóm 1: {team_1_count}"
        )

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Dự đoán nhãn cụm cho danh sách crops hình ảnh.

        Args:
            crops: Danh sách các crop hình ảnh

        Returns:
            np.ndarray: Nhãn cụm được dự đoán
        """
        if len(crops) == 0:
            logger.warning("Không có crops nào để dự đoán!")
            return np.array([])

        logger.debug(f"Dự đoán cho {len(crops)} crops")
        data = self.extract_features(crops)

        if data.size == 0:
            logger.error("Không trích xuất được đặc trưng để dự đoán!")
            return np.array([])

        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)
