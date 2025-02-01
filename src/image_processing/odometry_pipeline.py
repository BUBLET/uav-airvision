import logging
import cv2
import numpy as np
import os
from typing import Tuple, List, Optional

from .feature_extraction import FeatureExtractor
from .odometry_calculation import OdometryCalculator
from .frame_processor import FrameProcessor
from .trajectory_writer import TrajectoryWriter

class OdometryPipeline:
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 odometry_calculator: OdometryCalculator,
                 frame_processor: FrameProcessor,
                 logger: logging.Logger,
                 trajectory_writer: TrajectoryWriter):
        self.feature_extractor = feature_extractor
        self.odometry_calculator = odometry_calculator
        self.frame_processor = frame_processor
        self.logger = logger
        self.trajectory_writer = trajectory_writer
        # Абсолютная поза
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

    def run(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Не удалось открыть видео.")
            return
        ret, first_frame = cap.read()
        if not ret:
            self.logger.error("Первый кадр отсутствует.")
            cap.release()
            return
        # Инициализируем frame_processor
        self.frame_processor.process_frame(first_frame)  # это установит prev_gray и prev_pts
        # Записываем начальную позу
        self.trajectory_writer.write_pose(self.t_total, self.R_total)
        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.info("Достигнут конец видео.")
                break
            frame_idx += 1
            result = self.frame_processor.process_frame(frame)
            if result is None:
                self.logger.warning(f"Кадр {frame_idx} не обработан.")
                continue
            current_gray, current_pts, (R, t) = result
            # Аккумулируем движение:
            self.t_total += self.R_total @ t
            self.R_total = R @ self.R_total
            self.logger.info(f"Кадр {frame_idx}: t_total = {self.t_total.flatten()}, R_total =\n{self.R_total}")
            self.trajectory_writer.write_pose(self.t_total, self.R_total)
        cap.release()
        self.trajectory_writer.close()
        self.logger.info("Обработка завершена.")
