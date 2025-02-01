import logging
import cv2
import numpy as np
import os
from typing import Tuple

import config
from image_processing import FeatureExtractor
from image_processing import OdometryCalculator
from image_processing import FrameProcessor
from image_processing import TrajectoryWriter
from image_processing.odometry_pipeline import OdometryPipeline

def configure_logging() -> Tuple[logging.Logger, logging.Logger]:
    logger = logging.getLogger("OdometryPipeline")
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    runtime_handler = logging.FileHandler("logs/runtime.log", mode='w')
    runtime_handler.setLevel(logging.INFO)
    runtime_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    runtime_handler.setFormatter(runtime_formatter)
    logger.addHandler(runtime_handler)
    warning_handler = logging.FileHandler("logs/error.log", mode="w")
    warning_handler.setLevel(logging.WARNING)
    warning_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    warning_handler.setFormatter(warning_formatter)
    logger.addHandler(warning_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    metrics_logger = logging.getLogger("metrics_logger")
    metrics_logger.setLevel(logging.INFO)
    metrics_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    metrics_file_handler = logging.FileHandler("logs/metrics.log", mode='w')
    metrics_file_handler.setFormatter(metrics_formatter)
    metrics_logger.addHandler(metrics_file_handler)
    return logger, metrics_logger

def initialize_components(frame_width: int, frame_height: int) -> Tuple[FeatureExtractor, OdometryCalculator, FrameProcessor]:
    # Инициализируем FeatureExtractor
    feature_extractor = FeatureExtractor(
        grid_size=config.KPTS_UNIFORM_SELECTION_GRID_SIZE,
        max_pts_per_cell=config.MAX_PTS_PER_GRID,
        nfeatures=config.N_FEATURES,
        scaleFactor=config.INIT_SCALE,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=config.SCORE_TYPE,
        patchSize=config.PATCH_SIZE,
        fastThreshold=config.FAST_THRESHOLD
    )
    # OdometryCalculator
    odometry_calculator = OdometryCalculator(
        image_width=frame_width,
        image_height=frame_height,
        camera_matrix=config.CAMERA_MATRIX,
        e_ransac_threshold=config.E_RANSAC_THRESHOLD
    )
    # Настройка параметров optical flow (lk_params)
    lk_params = dict(winSize=config.LK_WIN_SIZE,
                     criteria=config.LK_CRITERIA)
    # FrameProcessor
    frame_processor = FrameProcessor(feature_extractor, odometry_calculator, lk_params)
    return feature_extractor, odometry_calculator, frame_processor

def main():
    logger, metrics_logger = configure_logging()
    video_path = config.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видео.")
        return
    ret, reference_frame = cap.read()
    if not ret:
        logger.error("Первый кадр отсутствует.")
        cap.release()
        return
    frame_height, frame_width = reference_frame.shape[:2]
    cap.release()
    feature_extractor, odometry_calculator, frame_processor = initialize_components(frame_width, frame_height)
    trajectory_writer = TrajectoryWriter("results/estimated_traj.txt")
    pipeline = OdometryPipeline(feature_extractor, odometry_calculator, frame_processor, logger, trajectory_writer)
    pipeline.run(video_path)

if __name__ == "__main__":
    main()
