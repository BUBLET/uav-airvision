import logging
import numpy as np
from typing import Tuple, List, Optional

import config
from .feature_extraction import FeatureExtractor
from .odometry_calculation import OdometryCalculator
from .frame_processor import FrameProcessor
from .trajectory_writer import TrajectoryWriter
from .kalman_filter import VIOFilter
from .imu_synchronizer import IMUSynchronizer

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


class OdometryPipeline:
    """
    VIO-пайплайн: объединяет VO и IMU в скользящем окне оптимизации + EKF-предсказание.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        odometry_calculator: OdometryCalculator,
        frame_processor: FrameProcessor,
        imu_synchronizer: IMUSynchronizer,
        logger: logging.Logger,
        trajectory_writer: TrajectoryWriter,
        window_size: int = config.WINDOW_SIZE
    ):
        # компоненты
        self.feature_extractor = feature_extractor
        self.odometry_calculator = odometry_calculator
        self.frame_processor = frame_processor
        self.imu_sync = imu_synchronizer
        self.logger = logger
        self.trajectory_writer = trajectory_writer

        # параметры окна оптимизации
        self.window_size = window_size
        self.window_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self.window_relatives: List[Tuple[np.ndarray, np.ndarray]] = []

        # глобальная поза
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

        # EKF-фильтр (VIOFilter)
        self.dt = 1.0 / config.VO_FPS
        self.vio_filter = VIOFilter(
            dt=self.dt,
            accel_noise=config.IMU_ACCEL_NOISE,
            vo_noise=config.VO_NOISE
        )
        logger.info(f"VIOFilter инициализирован: dt={self.dt}, "
                    f"accel_noise={config.IMU_ACCEL_NOISE}, vo_noise={config.VO_NOISE}")


    def pose_to_vector(self, R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        rotvec = R.from_matrix(R_mat).as_rotvec()
        return np.concatenate((rotvec, t_vec.flatten()))

    def vector_to_pose(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotvec = x[:3]
        t_vec = x[3:].reshape(3, 1)
        R_mat = R.from_rotvec(rotvec).as_matrix()
        return R_mat, t_vec

    def compose_poses(self, p1: Tuple[np.ndarray, np.ndarray],
                      p2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        R1, t1 = p1
        R2, t2 = p2
        return R1 @ R2, t1 + R1 @ t2

    def invert_pose(self, pose: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        Rm, tv = pose
        Ri = Rm.T
        ti = -Ri @ tv
        return Ri, ti

    def relative_error(self, x: np.ndarray) -> np.ndarray:
        N = len(self.window_poses)
        poses = [self.window_poses[0]]
        for i in range(1, N):
            xi = x[(i - 1) * 6 : i * 6]
            poses.append(self.vector_to_pose(xi))

        errs = []
        for i in range(N - 1):
            Ri, ti = poses[i]
            Rpc, tpc = poses[i+1]
            inv_R, inv_t = self.invert_pose((Ri, ti))
            R_pred, t_pred = self.compose_poses((inv_R, inv_t), (Rpc, tpc))
            R_meas, t_meas = self.window_relatives[i]
            errs.extend(R.from_matrix(R_meas.T @ R_pred).as_rotvec().tolist())
            errs.extend((t_pred - t_meas).flatten().tolist())

        lam = config.LAMBDA_REG
        for i in range(1, N - 1):
            p_prev = self.pose_to_vector(*poses[i-1])
            p_curr = self.pose_to_vector(*poses[i])
            p_next = self.pose_to_vector(*poses[i+1])
            second_diff = (p_next - p_curr) - (p_curr - p_prev)
            errs.extend((lam * second_diff).tolist())

        return np.array(errs)

    def optimize_window(self):
        N = len(self.window_poses)
        if N < 2:
            return
        x0 = np.concatenate([self.pose_to_vector(*self.window_poses[i]) for i in range(1, N)])
        self.logger.info(f"Оптимизирую окно из {N} кадров...")
        res = least_squares(
            self.relative_error, x0,
            verbose=1, xtol=1e-2, ftol=1e-2,
            loss='huber', f_scale=1.0
        )
        x_opt = res.x
        new_poses = [self.window_poses[0]]
        for i in range(1, N):
            xi = x_opt[(i - 1)*6 : i*6]
            new_poses.append(self.vector_to_pose(xi))
        self.window_poses = new_poses
        self.R_total, self.t_total = self.window_poses[-1]
        self.logger.info(f"Окно оптимизировано за {res.nfev} итераций")

    def run(self, loader):
        """
        loader должно быть экземпляром BaseDatasetLoader:
        он возвращает (frame, timestamp).
        """
        # читаем первый кадр
        first = loader.read_next()
        if first is None:
            self.logger.error("Пустой источник данных")
            return
        frame, ts = first

        # инициализируем EKF состоянием в нуле
        # (предсказание не делаем для первого кадра)
        self.trajectory_writer.write_pose(ts, self.t_total, self.R_total)
        self.window_poses.append((self.R_total.copy(), self.t_total.copy()))

        idx = 1
        while True:
            item = loader.read_next()
            if item is None:
                self.logger.info("Данные кончились")
                break
            frame, ts = item
            idx += 1

            # IMU-predict
            gyro, accel = self.imu_sync.get_measurements_for_frame(ts)
            self.vio_filter.predict(accel)

            # VO
            res = self.frame_processor.process_frame(frame)
            if res is None:
                self.logger.warning(f"Кадр {idx} пропущен")
                continue
            _, _, (R_rel, t_rel) = res

            # EKF-update
            self.vio_filter.update(t_rel)

            # получаем скорректированное положение
            pos, vel = self.vio_filter.get_state()
            self.t_total = pos.reshape(3,1)
            # для ориентации по-прежнему накапливаем VO
            self.R_total = R_rel @ self.R_total

            # окно для оптимизации
            self.window_relatives.append((R_rel.copy(), t_rel.copy()))
            self.window_poses.append((self.R_total.copy(), self.t_total.copy()))
            if len(self.window_poses) > self.window_size:
                self.optimize_window()
                self.window_poses = [(self.R_total.copy(), self.t_total.copy())]
                self.window_relatives = []

            self.logger.info(f"Кадр {idx}: t_total={self.t_total.flatten()}")
            self.trajectory_writer.write_pose(ts, self.t_total, self.R_total)

        # сохранение и завершение
        self.trajectory_writer.close()
        self.logger.info("Обработка закончена")
