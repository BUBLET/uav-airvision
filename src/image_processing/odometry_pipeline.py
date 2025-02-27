import logging
import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
import config

from .feature_extraction import FeatureExtractor
from .odometry_calculation import OdometryCalculator
from .frame_processor import FrameProcessor
from .trajectory_writer import TrajectoryWriter

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class OdometryPipeline:
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 odometry_calculator: OdometryCalculator,
                 frame_processor: FrameProcessor,
                 logger: logging.Logger,
                 trajectory_writer: TrajectoryWriter,
                 window_size = config.WINDOW_SIZE):  # колво кадров в окне
        self.feature_extractor = feature_extractor
        self.odometry_calculator = odometry_calculator
        self.frame_processor = frame_processor
        self.logger = logger
        self.trajectory_writer = trajectory_writer
        self.window_size = window_size

        self.R_total = np.eye(3)  # глобальный поворот стартуем с единичной матрицы
        self.t_total = np.zeros((3, 1))  # глобальный перенос стартуем с нуля

        # окно для оптимизации храним абсолютные позы и относительные переходы
        self.window_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self.window_relatives: List[Tuple[np.ndarray, np.ndarray]] = []

    def pose_to_vector(self, R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        # переводим R и t в вектор из 6 чисел axisangle и перенос
        rotvec = R.from_matrix(R_mat).as_rotvec()
        return np.concatenate((rotvec, t_vec.flatten()))

    def vector_to_pose(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # переводим вектор обратно в R и t
        rotvec = x[:3]
        t_vec = x[3:].reshape(3, 1)
        R_mat = R.from_rotvec(rotvec).as_matrix()
        return R_mat, t_vec

    def compose_poses(self, pose1: Tuple[np.ndarray, np.ndarray],
                      pose2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # соединяем две позы сначала R1 потом R2 а перенос суммируем
        R1, t1 = pose1
        R2, t2 = pose2
        return R1 @ R2, t1 + R1 @ t2

    def invert_pose(self, pose: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # переворачиваем позу получаем R транспонированную и -R^T*t
        R_mat, t_vec = pose
        R_inv = R_mat.T
        t_inv = -R_inv @ t_vec
        return R_inv, t_inv

    def relative_error(self, x: np.ndarray) -> np.ndarray:
        N = len(self.window_poses)
        poses = [self.window_poses[0]]
        for i in range(1, N):
            xi = x[(i - 1) * 6 : i * 6]
            poses.append(self.vector_to_pose(xi))
        
        errors = []
        for i in range(N - 1):
            inv_pose = self.invert_pose(poses[i])
            pred_pose = self.compose_poses(inv_pose, poses[i + 1])
            R_pred, t_pred = pred_pose
            R_meas, t_meas = self.window_relatives[i]
            errors.extend(R.from_matrix(R_meas.T @ R_pred).as_rotvec().tolist())
            errors.extend((t_pred - t_meas).flatten().tolist())
        
        lambda_reg = config.LAMBDA_REG  # коэффициент регуляризации, подбирается экспериментально
        for i in range(1, N - 1):
            pose_prev = self.pose_to_vector(*poses[i - 1])
            pose_curr = self.pose_to_vector(*poses[i])
            pose_next = self.pose_to_vector(*poses[i + 1])
            # Вторая разность
            second_diff = (pose_next - pose_curr) - (pose_curr - pose_prev)
            errors.extend((lambda_reg * second_diff).tolist())
        
        return np.array(errors)


    def optimize_window(self):
        N = len(self.window_poses)
        if N < 2:
            return
        x0 = []
        for i in range(1, N):
            x0.append(self.pose_to_vector(*self.window_poses[i]))
        x0 = np.concatenate(x0)
        self.logger.info("Оптимизирую окно с {} кадрами".format(N))
        res = least_squares(self.relative_error, x0, verbose=1, xtol=1e-2, ftol=1e-2, loss='huber', f_scale=1.0)
        x_opt = res.x
        new_window_poses = [self.window_poses[0]]  # первая поза фиксирована
        for i in range(1, N):
            xi = x_opt[(i - 1) * 6 : i * 6]
            new_window_poses.append(self.vector_to_pose(xi))
        self.window_poses = new_window_poses
        self.R_total, self.t_total = self.window_poses[-1]
        self.logger.info("Окно оптимизировано за {} итераций".format(res.nfev))


    def run(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Видео не открылось")
            return
        ret, first_frame = cap.read()
        if not ret:
            self.logger.error("Первый кадр не прочитал")
            cap.release()
            return
        init_result = self.frame_processor.process_frame(first_frame)
        if init_result is None:
            self.logger.error("Ошибка первого кадра")
            cap.release()
            return
        _, init_pts, _ = init_result
        self.trajectory_writer.write_pose(self.t_total, self.R_total)
        self.window_poses.append((self.R_total.copy(), self.t_total.copy()))
        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.info("Видео кончилось")
                break
            frame_idx += 1
            result = self.frame_processor.process_frame(frame)
            if result is None:
                self.logger.warning("Кадр {} пропустили".format(frame_idx))
                continue
            current_gray, current_pts, (R_rel, t_rel) = result
            # обновляем глобальную позу через накопление
            self.t_total = self.t_total + self.R_total @ t_rel
            self.R_total = R_rel @ self.R_total
            self.window_relatives.append((R_rel.copy(), t_rel.copy()))
            self.window_poses.append((self.R_total.copy(), self.t_total.copy()))
            if len(self.window_poses) > self.window_size:
                self.optimize_window()
                # сбрасываем окно оставляя только последнюю оптимизированную позу
                self.window_poses = [(self.R_total.copy(), self.t_total.copy())]
                self.window_relatives = []
            self.logger.info("Кадр {} t_total {} R_total".format(frame_idx, self.t_total.flatten()))
            self.trajectory_writer.write_pose(self.t_total, self.R_total)
        cap.release()
        self.trajectory_writer.close()
        self.logger.info("Обработка закончена")
