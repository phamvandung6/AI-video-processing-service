import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import cv2
import logging
import numpy as np
import supervision as sv
from tqdm import tqdm
from config import *
from utils1 import get_crops, resolve_goalkeepers_team_id, render_radar
from ultralytics import YOLO
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
import torch
from torch import amp
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import os
import threading
import pkg_resources

def validate_inputs(source_video_path: str, target_video_path: str, device: str) -> None:
    """Validate input paths and device availability."""
    if not os.path.exists(source_video_path):
        raise FileNotFoundError(f"Source video {source_video_path} does not exist.")
    if not os.access(os.path.dirname(target_video_path) or '.', os.W_OK):
        raise PermissionError(f"Cannot write to {target_video_path} directory.")
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

def run_yolo_batch(model, frames, imgsz):
    """Run YOLO model on a batch of frames with mixed precision."""
    try:
        with amp.autocast('cuda'):
            results = model(frames, imgsz=imgsz, verbose=False)
        return [sv.Detections.from_ultralytics(r) for r in results]
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA out of memory. Reducing batch size and retrying...")
            return run_yolo_batch(model, frames[:len(frames)//2], imgsz) + run_yolo_batch(model, frames[len(frames)//2:], imgsz)
        raise e

def write_frames(sink, frame_queue):
    """Write frames to VideoSink from a queue."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        sink.write_frame(frame)
    frame_queue.task_done()

def collect_crops_batch(source_path: str, model, max_frames: int = 100) -> list:
    """Collect crops in batches for team classification."""
    frames = []
    frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=STRIDE)
    for i, frame in enumerate(tqdm(frame_generator, desc='Collecting crops', total=max_frames)):
        if i >= max_frames:
            break
        frames.append(frame)
    crops = []
    batch_size = 6  # Increased for RTX 4060
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        results = run_yolo_batch(model, batch, imgsz=640)
        for frame, result in zip(batch, results):
            detections = result.with_nms(threshold=0.1)
            crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    return crops

def run_full_pipeline(source_video_path: str, target_video_path: str, device: str) -> None:
    """Run the full soccer video analysis pipeline."""
    # Validate inputs
    validate_inputs(source_video_path, target_video_path, device)

    # Initialize YOLO models for players and ball detection only
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # Fuse models to optimize performance (if needed)
    player_detection_model.model.fuse()
    ball_detection_model.model.fuse()

    # Collect crops for team classification
    crops = collect_crops_batch(source_video_path, player_detection_model)
    print(f"Number of crops collected: {len(crops)}")
    if len(crops) == 0:
        raise ValueError("No crops were collected. Check player detection step.")

    # Team classification
    try:
        team_classifier = TeamClassifier(device=device, batch_size=64)
        logging.info("Starting team classifier fitting...")
        team_classifier.fit(crops)
        logging.info("Team classifier fitting completed.")
    except Exception as e:
        logging.error(f"Error in TeamClassifier.fit: {e}")
        raise

    # Ball tracking setup
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=320, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.1)
        return detections

    slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(320, 320), overlap_ratio_wh=None, overlap_wh=(64, 64))

    # Player tracking setup
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # Process video
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    batch_size = 6  

    with sv.VideoSink(target_video_path, video_info) as sink:
        frame_queue = Queue(maxsize=10)
        writer_thread = threading.Thread(target=write_frames, args=(sink, frame_queue))
        writer_thread.start()

        frames = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc='Processing video')):
                frames.append(frame)
                if len(frames) == batch_size:
                    # Parallel YOLO processing for player and ball detection only
                    player_future = executor.submit(run_yolo_batch, player_detection_model, frames, 1280)
                    ball_future = executor.submit(run_yolo_batch, ball_detection_model, frames, 320)
                    player_results = player_future.result()
                    ball_results = ball_future.result()

                    for frame, player_result, ball_result in zip(frames, player_results, ball_results):
                        annotated_frame = frame.copy()

                        # Player detection and annotation
                        detections = player_result.with_nms(threshold=0.1)
                        detections = tracker.update_with_detections(detections)
                        ball_detections = slicer(frame)
                        ball_detections = ball_tracker.update(ball_detections)
                        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

                        players = detections[detections.class_id == PLAYER_CLASS_ID]
                        crops = get_crops(annotated_frame, players)
                        players_team_id = team_classifier.predict(crops)

                        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
                        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

                        referees = detections[detections.class_id == REFEREE_CLASS_ID]
                        detections = sv.Detections.merge([players, goalkeepers, referees])
                        color_lookup = np.array(
                            players_team_id.tolist() + goalkeepers_team_id.tolist() + [REFEREE_CLASS_ID] * len(referees)
                        )
                        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

                        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)
                        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                            annotated_frame, detections, labels, custom_color_lookup=color_lookup
                        )

                        frame_queue.put(annotated_frame)

                    frames = []

            # Process remaining frames
            if frames:
                player_results = run_yolo_batch(player_detection_model, frames, 1280)
                ball_results = run_yolo_batch(ball_detection_model, frames, 320)
                for frame, player_result, ball_result in zip(frames, player_results, ball_results):
                    annotated_frame = frame.copy()
                    detections = player_result.with_nms(threshold=0.1)
                    detections = tracker.update_with_detections(detections)
                    ball_detections = slicer(frame)
                    ball_detections = ball_tracker.update(ball_detections)
                    annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

                    players = detections[detections.class_id == PLAYER_CLASS_ID]
                    crops = get_crops(annotated_frame, players)
                    players_team_id = team_classifier.predict(crops)

                    goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
                    goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

                    referees = detections[detections.class_id == REFEREE_CLASS_ID]
                    detections = sv.Detections.merge([players, goalkeepers, referees])
                    color_lookup = np.array(
                        players_team_id.tolist() + goalkeepers_team_id.tolist() + [REFEREE_CLASS_ID] * len(referees)
                    )
                    labels = [str(tracker_id) for tracker_id in detections.tracker_id]

                    annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)
                    annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                        annotated_frame, detections, labels, custom_color_lookup=color_lookup
                    )

                    frame_queue.put(annotated_frame)

        frame_queue.put(None)
        writer_thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer Video Analysis Pipeline')
    parser.add_argument('--source_video_path', type=str, required=True, help='Path to the source video')
    parser.add_argument('--target_video_path', type=str, required=True, help='Path to save the processed video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., cpu, cuda)')
    args = parser.parse_args()

    run_full_pipeline(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device
    )
