import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
import supervision as sv
from torch import amp
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VideoProcessor")

# Import từ config
from config import (
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    STRIDE,
)

# Import từ utilities
from utils1 import get_crops, resolve_goalkeepers_team_id


class VideoProcessor:
    """Lớp xử lý video sử dụng các mô hình YOLO và thuật toán phân loại đội."""

    def __init__(self, output_dir="output_videos"):
        """Khởi tạo bộ xử lý video

        Args:
            output_dir: Thư mục đầu ra cho video đã xử lý
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Đã khởi tạo VideoProcessor với thư mục đầu ra: {self.output_dir}")

    def validate_inputs(
        self, source_video_path: str, target_video_path: str, device: str
    ) -> None:
        """Kiểm tra đầu vào hợp lệ.

        Args:
            source_video_path: Đường dẫn đến video nguồn
            target_video_path: Đường dẫn đến video đích
            device: Thiết bị xử lý (cuda, cpu)

        Raises:
            FileNotFoundError: Nếu video nguồn không tồn tại
            PermissionError: Nếu không thể ghi vào thư mục đích
            RuntimeError: Nếu yêu cầu CUDA nhưng không có sẵn
        """
        if not os.path.exists(source_video_path):
            raise FileNotFoundError(f"Video nguồn {source_video_path} không tồn tại.")
        if not os.access(os.path.dirname(target_video_path) or ".", os.W_OK):
            raise PermissionError(f"Không thể ghi vào thư mục {target_video_path}.")

    def run_yolo_batch(self, model, frames, imgsz):
        """Chạy mô hình YOLO trên một batch frames với mixed precision.

        Args:
            model: Mô hình YOLO
            frames: Danh sách các frames
            imgsz: Kích thước ảnh đầu vào

        Returns:
            list: Danh sách các kết quả phát hiện
        """
        try:
            with amp.autocast("cuda"):
                results = model(frames, imgsz=imgsz, verbose=False)
            return [sv.Detections.from_ultralytics(r) for r in results]
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(
                    "CUDA out of memory. Reducing batch size and retrying..."
                )
                return self.run_yolo_batch(
                    model, frames[: len(frames) // 2], imgsz
                ) + self.run_yolo_batch(model, frames[len(frames) // 2 :], imgsz)
            raise e

    def write_frames(self, sink, frame_queue):
        """Ghi frames vào VideoSink từ queue.

        Args:
            sink: VideoSink để ghi frames
            frame_queue: Queue chứa frames cần ghi
        """
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            sink.write_frame(frame)
        frame_queue.task_done()

    def collect_crops_batch(
        self, source_path: str, model, max_frames: int = 100
    ) -> list:
        """Thu thập crops theo batch để phân loại đội.

        Args:
            source_path: Đường dẫn đến video nguồn
            model: Mô hình phát hiện đối tượng
            max_frames: Số frame tối đa để xử lý

        Returns:
            list: Danh sách các crops
        """
        frames = []
        frame_generator = sv.get_video_frames_generator(
            source_path=source_path, stride=STRIDE
        )
        for i, frame in enumerate(
            tqdm(frame_generator, desc="Collecting crops", total=max_frames)
        ):
            if i >= max_frames:
                break
            frames.append(frame)

        crops = []
        batch_size = 6  # Batch size phù hợp với GPUs hiện đại
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            results = self.run_yolo_batch(model, batch, imgsz=640)
            for frame, result in zip(batch, results):
                detections = result.with_nms(threshold=0.1)
                crops += get_crops(
                    frame, detections[detections.class_id == PLAYER_CLASS_ID]
                )

        logger.info(
            f"Đã thu thập {len(crops)} crops từ {min(len(frames), max_frames)} frames"
        )
        return crops

    def preserve_audio_from_original(
        self, original_video, processed_video, output_video
    ):
        """Trích xuất âm thanh từ video gốc và thêm vào video đã xử lý.

        Args:
            original_video (str): Đường dẫn đến video gốc có âm thanh
            processed_video (str): Đường dẫn đến video đã xử lý (không có âm thanh)
            output_video (str): Đường dẫn đến video đầu ra

        Returns:
            bool: True nếu thành công
        """
        try:
            temp_output = f"{output_video}.tmp.mp4"

            # Sử dụng FFmpeg để kết hợp video đã xử lý với âm thanh từ video gốc
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                processed_video,  # Video đã xử lý (không âm thanh)
                "-i",
                original_video,  # Video gốc (có âm thanh)
                "-c:v",
                "copy",  # Giữ nguyên stream video
                "-c:a",
                "aac",  # Mã hóa âm thanh với AAC
                "-map",
                "0:v:0",  # Lấy stream video từ đầu vào thứ nhất
                "-map",
                "1:a:0",  # Lấy stream âm thanh từ đầu vào thứ hai
                "-shortest",  # Kết thúc khi stream ngắn nhất kết thúc
                temp_output,
            ]

            logger.info(
                f"Đang thêm âm thanh vào video đã xử lý: {original_video} + {processed_video} -> {output_video}"
            )
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if process.returncode != 0:
                logger.error(f"Lỗi khi thêm âm thanh: {process.stderr}")
                return False

            # Di chuyển file kết quả
            if os.path.exists(temp_output):
                if os.path.exists(output_video):
                    os.remove(output_video)
                os.rename(temp_output, output_video)
                logger.info(f"Đã thêm âm thanh thành công: {output_video}")
                return True
            else:
                logger.error(f"File kết quả không được tạo: {temp_output}")
                return False

        except Exception as e:
            logger.error(f"Lỗi khi thêm âm thanh: {e}")
            return False

    def process_soccer_video(
        self,
        source_video_path: str,
        target_video_path: str,
        player_detection_model,
        ball_detection_model,
        team_classifier,
        device: str = "cuda",
        progress_callback=None,
    ):
        """Xử lý video bóng đá đầy đủ.

        Args:
            source_video_path: Đường dẫn đến video nguồn
            target_video_path: Đường dẫn đến video đích
            player_detection_model: Mô hình phát hiện cầu thủ
            ball_detection_model: Mô hình phát hiện bóng
            team_classifier: Bộ phân loại đội
            device: Thiết bị xử lý (cuda, cpu)
            progress_callback: Callback để cập nhật tiến độ (0-100%)

        Returns:
            bool: True nếu xử lý thành công
        """
        # Import những gì cần từ các thư viện
        from sports.common.ball import BallAnnotator, BallTracker

        # Kiểm tra đầu vào
        self.validate_inputs(source_video_path, target_video_path, device)

        # Tạo file trung gian cho video xử lý (không có âm thanh)
        processing_path = target_video_path + ".processing.mp4"

        # Thiết lập ball tracking
        ball_tracker = BallTracker(buffer_size=20)
        ball_annotator = BallAnnotator(radius=6, buffer_size=10)
        slicer = sv.InferenceSlicer(
            callback=lambda image_slice: sv.Detections.from_ultralytics(
                ball_detection_model(image_slice, imgsz=320, verbose=False)[0]
            ).with_nms(threshold=0.1),
            slice_wh=(320, 320),
            overlap_ratio_wh=None,
            overlap_wh=(64, 64),
        )

        # Thiết lập tracker
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)

        # Thu thập crops để phân loại đội
        logger.info("Thu thập crops để phân loại đội...")
        crops = self.collect_crops_batch(source_video_path, player_detection_model)
        if len(crops) == 0:
            raise ValueError(
                "Không thu thập được crops. Kiểm tra bước phát hiện cầu thủ."
            )

        # Phân loại đội
        try:
            logger.info("Bắt đầu huấn luyện bộ phân loại đội...")
            team_classifier.fit(crops)
            logger.info("Đã hoàn thành huấn luyện bộ phân loại đội.")
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện bộ phân loại đội: {e}")
            raise

        # Xử lý video
        logger.info(f"Bắt đầu xử lý video chính: {source_video_path}")

        # Lấy thông tin video và chuẩn bị generator
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        batch_size = 6

        # Tạo video sink
        with sv.VideoSink(processing_path, video_info) as sink:
            frame_queue = Queue(maxsize=10)
            writer_thread = threading.Thread(
                target=self.write_frames, args=(sink, frame_queue)
            )
            writer_thread.start()

            frames = []
            total_frames = video_info.total_frames
            processed_frames = 0

            with ThreadPoolExecutor(max_workers=2) as executor:
                for i, frame in enumerate(
                    tqdm(frame_generator, total=total_frames, desc="Processing video")
                ):
                    frames.append(frame)
                    if len(frames) == batch_size:
                        # Xử lý song song cho phát hiện cầu thủ và bóng
                        player_future = executor.submit(
                            self.run_yolo_batch, player_detection_model, frames, 1280
                        )
                        ball_future = executor.submit(
                            self.run_yolo_batch, ball_detection_model, frames, 320
                        )
                        player_results = player_future.result()
                        ball_results = ball_future.result()

                        for frame, player_result, ball_result in zip(
                            frames, player_results, ball_results
                        ):
                            annotated_frame = self._process_frame(
                                frame,
                                player_result,
                                ball_result,
                                tracker,
                                ball_tracker,
                                ball_annotator,
                                slicer,
                                team_classifier,
                            )
                            frame_queue.put(annotated_frame)

                            # Cập nhật tiến độ
                            processed_frames += 1
                            if progress_callback:
                                progress_percentage = (
                                    processed_frames / total_frames
                                ) * 100
                                progress_callback(progress_percentage)

                        frames = []

                # Xử lý frames còn lại
                if frames:
                    player_results = self.run_yolo_batch(
                        player_detection_model, frames, 1280
                    )
                    ball_results = self.run_yolo_batch(
                        ball_detection_model, frames, 320
                    )
                    for frame, player_result, ball_result in zip(
                        frames, player_results, ball_results
                    ):
                        annotated_frame = self._process_frame(
                            frame,
                            player_result,
                            ball_result,
                            tracker,
                            ball_tracker,
                            ball_annotator,
                            slicer,
                            team_classifier,
                        )
                        frame_queue.put(annotated_frame)

                        # Cập nhật tiến độ
                        processed_frames += 1
                        if progress_callback:
                            progress_percentage = (
                                processed_frames / total_frames
                            ) * 100
                            progress_callback(progress_percentage)

            # Báo hiệu kết thúc và đợi thread ghi hình hoàn thành
            frame_queue.put(None)
            writer_thread.join()

        # Thêm âm thanh từ video gốc vào video đã xử lý
        logger.info("Thêm âm thanh từ video gốc vào video đã xử lý...")
        audio_success = self.preserve_audio_from_original(
            source_video_path, processing_path, target_video_path
        )

        if not audio_success:
            logger.warning(
                "Không thể thêm âm thanh, sử dụng video đã xử lý không có âm thanh."
            )
            # Sao chép file processing sang target nếu không thành công khi thêm âm thanh
            import shutil

            shutil.copy2(processing_path, target_video_path)

        # Xóa file trung gian nếu có
        if os.path.exists(processing_path):
            os.remove(processing_path)

        logger.info(
            f"Đã hoàn thành xử lý video: {source_video_path} -> {target_video_path}"
        )
        return True

    def _process_frame(
        self,
        frame,
        player_result,
        ball_result,
        tracker,
        ball_tracker,
        ball_annotator,
        slicer,
        team_classifier,
    ):
        """Xử lý một frame đơn lẻ.

        Args:
            frame: Frame cần xử lý
            player_result: Kết quả phát hiện cầu thủ
            ball_result: Kết quả phát hiện bóng
            tracker: Tracker theo dõi đối tượng
            ball_tracker: Tracker dành riêng cho bóng
            ball_annotator: Annotator cho bóng
            slicer: Inference slicer
            team_classifier: Bộ phân loại đội

        Returns:
            np.ndarray: Frame đã xử lý
        """
        annotated_frame = frame.copy()

        # Xử lý phát hiện cầu thủ
        detections = player_result.with_nms(threshold=0.1)
        detections = tracker.update_with_detections(detections)

        # Xử lý phát hiện bóng
        ball_detections = slicer(frame)
        ball_detections = ball_tracker.update(ball_detections)
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

        # Trích xuất và phân loại cầu thủ
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(annotated_frame, players)
        players_team_id = team_classifier.predict(crops)

        # Trích xuất thủ môn và trọng tài
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Gộp các phát hiện và chuẩn bị màu sắc
        merged_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )

        # Chuẩn bị nhãn
        labels = [str(tracker_id) for tracker_id in merged_detections.tracker_id]

        # Vẽ annotations
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, merged_detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, merged_detections, labels, custom_color_lookup=color_lookup
        )

        return annotated_frame

    def compress_video(self, input_path, output_path, crf=23, preset="medium"):
        """Nén video với ffmpeg giữ chất lượng tốt nhưng giảm dung lượng.

        Args:
            input_path (str): Đường dẫn của video input
            output_path (str): Đường dẫn của video output (có thể trùng với input_path)
            crf (int): Hệ số chất lượng (18-28, thấp = chất lượng cao)
            preset (str): Preset nén (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)

        Returns:
            bool: True nếu nén thành công
        """
        try:
            temp_output = f"{output_path}.tmp.mp4"

            # Sử dụng ffmpeg để nén video
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                temp_output,
            ]

            logger.info(f"Đang nén video: {input_path}")
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if process.returncode != 0:
                logger.error(f"Lỗi khi nén video: {process.stderr}")
                return False

            # Nếu đầu ra trùng với đầu vào, xóa file đầu vào
            if os.path.exists(temp_output):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_output, output_path)

                # Kiểm tra kích thước file
                original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
                compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                reduction = 100 * (1 - compressed_size / original_size)

                logger.info(f"Đã nén video thành công: {input_path} -> {output_path}")
                logger.info(
                    f"Kích thước giảm: {original_size:.2f}MB -> {compressed_size:.2f}MB ({reduction:.2f}%)"
                )
                return True
            else:
                logger.error(f"File nén không được tạo: {temp_output}")
                return False

        except Exception as e:
            logger.error(f"Lỗi khi nén video: {e}")
            return False
