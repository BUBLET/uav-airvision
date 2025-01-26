import logging
import cv2
import numpy as np
import os
from typing import Tuple

import config

from image_processing import FeatureExtractor
from image_processing import FeatureMatcher
from image_processing import OdometryCalculator
from image_processing import FrameProcessor
from image_processing import TrajectoryWriter
from image_processing.odometry_pipeline import OdometryPipeline


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

    # Handler для warning и error логов
    warning_handler = logging.FileHandler("logs/error.log", mode="w")
    warning_handler.setLevel(logging.WARNING)
    warning_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    warning_handler.setFormatter(warning_formatter)
    logger.addHandler(warning_handler)

    # Logger для метрик
    metrics_logger = logging.getLogger("metrics_logger")
    metrics_logger.setLevel(logging.INFO)
    metrics_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    metrics_file_handler = logging.FileHandler("logs/metrics.log", mode='w')
    metrics_file_handler.setFormatter(metrics_formatter)
    metrics_logger.addHandler(metrics_file_handler)

    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger, metrics_logger


def initialize_components(
    # Параметры для FeatureExtractor
    grid_size: int,
    max_pts_per_cell: int,
    nfeatures: int,
    scaleFactor: float,
    nlevels: int,
    edgeThreshold: int,
    firstLevel: int,
    WTA_K: int,
    scoreType: int,
    patchSize: int,
    fastThreshold: int,

    # Параметры для FeatureMatcher
    knn_k: int,
    lowe_ratio: float,
    norm_type: int,

    # Параметры для OdometryCalculator
    camera_matrix: np.ndarray,
    e_ransac_threshold: float,
    h_ransac_threshold: float,
    distance_threshold: float,
    map_clean_max_distance: float,
    reprojection_threshold: float,
    ratio_thresh: float,
    dist_coeffs: np.ndarray,
    ckd_radius: float,

    # Параметры для FrameProcessor
    translation_threshold: float,
    rotation_threshold: float,
    triangulation_threshold: float,
    bundle_adjustment_frames: int,
    force_keyframe_interval: int,
    homography_inlier_ratio: float,

    frame_width: int,
    frame_height: int
) -> Tuple[FeatureExtractor, FeatureMatcher, OdometryCalculator, FrameProcessor]:
    """
    Инициализирует все необходимые компоненты системы.
    """
    feature_extractor = FeatureExtractor(
        grid_size=grid_size,
        max_pts_per_cell=max_pts_per_cell,
        nfeatures=nfeatures,
        scaleFactor=scaleFactor,
        nlevels=nlevels,
        edgeThreshold=edgeThreshold,
        firstLevel=firstLevel,
        WTA_K=WTA_K,
        scoreType=scoreType,
        patchSize=patchSize,
        fastThreshold=fastThreshold
    )

    feature_matcher = FeatureMatcher(
        knn_k=knn_k,
        lowe_ratio=lowe_ratio,
        norm_type=norm_type
    )

    odometry_calculator = OdometryCalculator(
        camera_matrix=camera_matrix,
        e_ransac_threshold=e_ransac_threshold,
        h_ransac_threshold=h_ransac_threshold,
        distance_threshold=distance_threshold,
        map_clean_max_distance=map_clean_max_distance,
        reprojection_threshold=reprojection_threshold,
        ratio_thresh=ratio_thresh,
        dist_coeffs=dist_coeffs,
        ckd_radius=ckd_radius,
        image_width=frame_width,
        image_height=frame_height
    )

    processor = FrameProcessor(
        feature_extractor=feature_extractor,
        feature_matcher=feature_matcher,
        odometry_calculator=odometry_calculator,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold,
        triangulation_threshold=triangulation_threshold,
        bundle_adjustment_frames=bundle_adjustment_frames,
        force_keyframe_interval=force_keyframe_interval,
        homography_inlier_ratio=homography_inlier_ratio
    )

    return feature_extractor, feature_matcher, odometry_calculator, processor


def main():
    logger, metrics_logger = configure_logging()

    # Параметры для FeatureExtractor
    grid_size = config.KPTS_UNIFORM_SELECTION_GRID_SIZE
    max_pts_per_cell = config.MAX_PTS_PER_GRID
    nfeatures = config.N_FEATURES
    scaleFactor = config.INIT_SCALE
    nlevels = config.N_LEVELS
    edgeThreshold = config.EDGE_THRESHOLD
    firstLevel = 0
    WTA_K = 3
    scoreType = cv2.ORB_HARRIS_SCORE
    patchSize = 31
    fastThreshold = 10

    # Параметры для FeatureMatcher
    knn_k = 2
    lowe_ratio = config.LOWE_RATIO
    norm_type = cv2.NORM_HAMMING

    # Параметры для OdometryCalculator
    camera_matrix = config.CAMERA_MATRIX
    e_ransac_threshold = config.E_RANSAC_THRESHOLD
    h_ransac_threshold = config.H_RANSAC_THRESHOLD
    distance_threshold = config.DISTANCE_THRESHOLD
    map_clean_max_distance = config.MAP_CLEAN_MAX_DISTANCE
    reprojection_threshold = config.REPROJECTION_THRESHOLD
    ratio_thresh = config.RATIO_THRESH
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ckd_radius = config.CKD_RADIUS

    # Параметры для FrameProcessor
    translation_threshold = config.TRANSLATION_THRESHOLD
    rotation_threshold = config.ROTATION_THRESHOLD
    triangulation_threshold = config.TRIANGULATION_THRESHOLD
    bundle_adjustment_frames = config.BUNDLE_ADJUSTMENT_FRAMES
    force_keyframe_interval = config.FORCE_KEYFRAME_INTERVAL
    homography_inlier_ratio = config.HOMOGRAPHY_INLIER_RATIO

    # Путь до видео и файл траектории
    video_path = config.VIDEO_PATH
    output_trajectory_path = "results/estimated_traj.txt"

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

    # Инициализация компонентов с прямой передачей параметров
    feature_extractor, feature_matcher, odometry_calculator, processor = initialize_components(
        # FeatureExtractor параметры
        grid_size,
        max_pts_per_cell,
        nfeatures,
        scaleFactor,
        nlevels,
        edgeThreshold,
        firstLevel,
        WTA_K,
        scoreType,
        patchSize,
        fastThreshold,

        # FeatureMatcher параметры
        knn_k,
        lowe_ratio,
        norm_type,

        # OdometryCalculator параметры
        camera_matrix,
        e_ransac_threshold,
        h_ransac_threshold,
        distance_threshold,
        map_clean_max_distance,
        reprojection_threshold,
        ratio_thresh,
        dist_coeffs,
        ckd_radius,

        # FrameProcessor параметры
        translation_threshold,
        rotation_threshold,
        triangulation_threshold,
        bundle_adjustment_frames,
        force_keyframe_interval,
        homography_inlier_ratio,

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
