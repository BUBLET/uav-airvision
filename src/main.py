import logging
import cv2
import numpy as np
import os
from typing import Tuple
import glob

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

class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Не удалось открыть видеофайл: " + video_path)
    
    def read_next(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        self.cap.release()

class FramesReader:
    def __init__(self, frames_folder, ext="png"):
        self.frames_paths = sorted(glob.glob(os.path.join(frames_folder, f"*.{ext}")))
        self.index = 0
        if not self.frames_paths:
            raise ValueError("Не найдено изображений в папке: " + frames_folder)
    
    def read_next(self):
        if self.index >= len(self.frames_paths):
            return None
        frame = cv2.imread(self.frames_paths[self.index])
        self.index += 1
        return frame
    
    def release(self):
        pass  

def configure_reading(source_type: str):
    if source_type == "video":
        return VideoReader(config.VIDEO_PATH)
    elif source_type == "frames":
        return FramesReader(config.FRAMES_FOLDER, ext="png")
    else:
        raise ValueError("Неверный тип источника: " + source_type)

def main():

    source_type = "video"  

    # Настройка логирования и начальных компонентов
    logger, _ = configure_logging()
    reader = configure_reading(source_type)

    # Получаем первый кадр для инициализации размеров
    first_frame = reader.read_next()
    if first_frame is None:
        logger.error("Не удалось получить первый кадр.")
        return

    frame_height, frame_width = first_frame.shape[:2]
    
    # Инициализация компонентов
    feature_extractor = FeatureExtractor(
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
    odometry_calculator = OdometryCalculator(
        image_width=frame_width,
        image_height=frame_height,
        camera_matrix=config.CAMERA_MATRIX,
        e_ransac_threshold=config.E_RANSAC_THRESHOLD
    )
    lk_params = dict(winSize=config.LK_WIN_SIZE,
                     criteria=config.LK_CRITERIA)
    frame_processor = FrameProcessor(feature_extractor, odometry_calculator, lk_params)
    trajectory_writer = TrajectoryWriter("results/estimated_traj.txt")
    pipeline = OdometryPipeline(feature_extractor, odometry_calculator, frame_processor, logger, trajectory_writer)

    # Обработка первого кадра
    init_result = frame_processor.process_frame(first_frame)
    if init_result is None:
        logger.error("Ошибка первого кадра")
        reader.release()
        return
    _, init_pts, _ = init_result
    pipeline.window_poses.append((pipeline.R_total.copy(), pipeline.t_total.copy()))
    trajectory_writer.write_pose(pipeline.t_total, pipeline.R_total)

    frame_idx = 1
    while True:
        frame = reader.read_next()
        if frame is None:
            logger.info("Истощились кадры")
            break
        frame_idx += 1
        result = frame_processor.process_frame(frame)
        if result is None:
            logger.warning("Кадр {} пропущен".format(frame_idx))
            continue
        current_gray, current_pts, (R_rel, t_rel) = result
        pipeline.t_total = pipeline.t_total + pipeline.R_total @ t_rel
        pipeline.R_total = R_rel @ pipeline.R_total
        pipeline.window_relatives.append((R_rel.copy(), t_rel.copy()))
        pipeline.window_poses.append((pipeline.R_total.copy(), pipeline.t_total.copy()))
        if len(pipeline.window_poses) > pipeline.window_size:
            pipeline.optimize_window()
            pipeline.window_poses = [(pipeline.R_total.copy(), pipeline.t_total.copy())]
            pipeline.window_relatives = []
        logger.info("Кадр {} t_total {} R_total".format(frame_idx, pipeline.t_total.flatten()))
        trajectory_writer.write_pose(pipeline.t_total, pipeline.R_total)
    reader.release()
    trajectory_writer.close()
    logger.info("Обработка закончена")

if __name__ == "__main__":
    main()