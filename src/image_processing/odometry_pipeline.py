# src/image_processing/odometry_pipeline.py

import logging
import cv2
import numpy as np
import os
from typing import Tuple, List, Optional

from .feature_extraction import FeatureExtractor
from .feature_matching import FeatureMatcher
from .odometry_calculation import OdometryCalculator
from .frame_processor import FrameProcessor
from .map_point import MapPoint
from .trajectory_writer import TrajectoryWriter


class OdometryPipeline:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        feature_matcher: FeatureMatcher,
        odometry_calculator: OdometryCalculator,
        frame_processor: FrameProcessor,
        logger: logging.Logger,
        metrics_logger: logging.Logger,
        lost_threshold: int,
        trajectory_writer: TrajectoryWriter
    ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.odometry_calculator = odometry_calculator
        self.frame_processor = frame_processor
        self.logger = logger
        self.metrics_logger = metrics_logger
        self.lost_threshold = lost_threshold
        self.trajectory_writer = trajectory_writer

        self.keyframes: List[Tuple[int, List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray]] = []
        self.poses: List[np.ndarray] = []
        self.map_points: List[MapPoint] = []
        self.lost_frames_count = 0
        self.initialization_completed = False

    def initialize(self, reference_frame: np.ndarray, frame_idx: int = 1):
        """
        Обрабатывает начальный кадр для инициализации системы.
        """
        ref_keypoints, ref_descriptors = self.feature_extractor.extract_features(reference_frame)
        if not ref_keypoints:
            raise ValueError("В опорном кадре не удалось найти ключевые точки.")

        initial_translation = np.zeros((3, 1), dtype=np.float32)
        initial_rotation = np.eye(3, dtype=np.float32)
        initial_pose = np.hstack((initial_rotation, initial_translation))
        self.keyframes.append((frame_idx, ref_keypoints, ref_descriptors, initial_pose))
        self.poses.append(initial_pose)

        self.trajectory_writer.write_pose(initial_pose)

    def process_frame(self, frame_idx: int, current_frame: np.ndarray):
        """
        Обрабатывает один кадр видео.
        """
        if not self.keyframes:
            self.logger.warning("Система не инициализирована. Пропуск кадра.")
            return

        result = self.frame_processor.process_frame(
            frame_idx,
            current_frame,
            self.keyframes[-1][1],  # ref_keypoints
            self.keyframes[-1][2],  # ref_descriptors
            self.poses[-1],         # last_pose
            self.map_points,
            self.initialization_completed,
            self.poses,
            self.keyframes
        )

        if result is None:
            self.lost_frames_count += 1
            self.logger.warning(f"Кадр {frame_idx} не обработан. Счётчик потерянных кадров: {self.lost_frames_count}")
            if self.lost_frames_count > self.lost_threshold:
                self.logger.warning(f"Сброс инициализации после {self.lost_threshold} потерянных кадров.")
                self._reset()
            self.metrics_logger.info(f"[LOST] Frame {frame_idx}, lost_frames_count={self.lost_frames_count}")
        else:
            self.lost_frames_count = 0
            ref_keypoints, ref_descriptors, last_pose, self.map_points, self.initialization_completed = result
            self.keyframes.append((frame_idx, ref_keypoints, ref_descriptors, last_pose))
            self.poses.append(last_pose)
            self.trajectory_writer.write_pose(last_pose)

    def _reset(self):
        """
        Сбрасывает состояние пайплайна.
        """
        self.initialization_completed = False
        self.map_points = []
        self.keyframes = []
        self.poses = []
        self.lost_frames_count = 0

    def run(self, video_path: str):
        """
        Запускает обработку видео.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Не удалось открыть видео.")
            return

        ret, reference_frame = cap.read()
        if not ret:
            self.logger.error("Первый кадр отсутствует.")
            cap.release()
            return

        try:
            self.initialize(reference_frame, frame_idx=1)
        except ValueError as e:
            self.logger.error(e)
            cap.release()
            return

        frame_idx = 1

        while True:
            ret, current_frame = cap.read()
            if not ret:
                self.logger.info("Достигнут конец видео.")
                break

            frame_idx += 1
            self.process_frame(frame_idx, current_frame)

        cap.release()
        self.trajectory_writer.close()  # Закрываем файл траектории
        cv2.destroyAllWindows()
        self.logger.info("Обработка завершена.")
