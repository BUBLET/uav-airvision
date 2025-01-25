import logging
import cv2
import numpy as np
import os
from typing import Tuple

from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator
from image_processing.frame_processor import FrameProcessor
from image_processing.trajectory_writer import TrajectoryWriter
from image_processing.odometry_pipeline import OdometryPipeline
from optimization.ba import BundleAdjustment


def configure_logging() -> Tuple[logging.Logger, logging.Logger]:
    """
    Настраивает логгирование для runtime и warning, а также для метрик.
    """
    logger = logging.getLogger("OdometryPipeline")
    logger.setLevel(logging.INFO)

    # Создание директории logs, если она не существует
    os.makedirs("logs", exist_ok=True)

    # Handler для runtime логов
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

    metrics_logger = logging.getLogger("metrics_logger")
    metrics_logger.setLevel(logging.INFO)
    metrics_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    metrics_file_handler = logging.FileHandler("logs/metrics.log", mode='w')
    metrics_file_handler.setFormatter(metrics_formatter)
    metrics_logger.addHandler(metrics_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger, metrics_logger


def initialize_components(
    feature_extractor_params: dict,
    feature_matcher_params: dict,
    odometry_calculator_params: dict,
    frame_processor_params: dict,
    frame_width: int,
    frame_height: int
) -> Tuple[FeatureExtractor, FeatureMatcher, OdometryCalculator, FrameProcessor]:
    """
    Инициализирует все необходимые компоненты системы.

    """
    feature_extractor = FeatureExtractor(**feature_extractor_params)
    feature_matcher = FeatureMatcher(**feature_matcher_params)
    odometry_calculator = OdometryCalculator(
        image_width=frame_width,
        image_height=frame_height,
        **odometry_calculator_params
    )
    processor = FrameProcessor(
        feature_extractor=feature_extractor,
        feature_matcher=feature_matcher,
        odometry_calculator=odometry_calculator,
        **frame_processor_params
    )
    return feature_extractor, feature_matcher, odometry_calculator, processor


def main():
    logger, metrics_logger = configure_logging()

    feature_extractor_params = {
        "grid_size": 48,
        "max_pts_per_cell": 2,
        "nfeatures": 90000,
        "scaleFactor": 1.2,
        "nlevels": 10,
        "edgeThreshold": 10,
        "firstLevel": 0,
        "WTA_K": 3,
        "scoreType": cv2.ORB_HARRIS_SCORE,
        "patchSize": 31,
        "fastThreshold": 30
    }

    feature_matcher_params = {
        "knn_k": 2,
        "lowe_ratio": 0.45,
        "norm_type": cv2.NORM_HAMMING,
    }

    odometry_calculator_params = {
        "camera_matrix": np.array([[615.0, 0, 320.0],
                                   [0, 615.0, 240.0],
                                   [0, 0, 1]], dtype=np.float64),
        "e_ransac_threshold": 0.6,
        "h_ransac_threshold": 0.1,
        "distance_threshold": 77,
        "map_clean_max_distance": 195,
        "reprojection_threshold": 4.72,
        "ratio_thresh": 0.45,
        "dist_coeffs": np.zeros((4, 1), dtype=np.float64),
        "ckd_radius": 5,
    }

    frame_processor_params = {
        "translation_threshold": 1.4,
        "rotation_threshold": np.deg2rad(2.75),
        "triangulation_threshold": np.deg2rad(2.0),
        "bundle_adjustment_frames": 12,
        "force_keyframe_interval": 1,
        "homography_inlier_ratio": 0.86,
    }

    # Путь до видео и файл траектории
    video_path = "D:/Мисис/Диплом/AirVision/datasets/output.mp4"
    output_trajectory_path = "results/estimated_traj.txt"

    # Создание директории results, если она не существует
    os.makedirs(os.path.dirname(output_trajectory_path), exist_ok=True)

    # Открытие видео для получения размеров первого кадра
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

    feature_extractor, feature_matcher, odometry_calculator, processor = initialize_components(
        feature_extractor_params,
        feature_matcher_params,
        odometry_calculator_params,
        frame_processor_params,
        frame_width,
        frame_height
    )

    trajectory_writer = TrajectoryWriter(output_trajectory_path)

    pipeline = OdometryPipeline(
        feature_extractor=feature_extractor,
        feature_matcher=feature_matcher,
        odometry_calculator=odometry_calculator,
        frame_processor=processor,
        logger=logger,
        metrics_logger=metrics_logger,
        lost_threshold=10,
        trajectory_writer=trajectory_writer
    )

    # Запуск пайплайна
    pipeline.run(video_path)


if __name__ == "__main__":
    main()
