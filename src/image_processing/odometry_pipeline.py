# src/image_processing/odometry_pipeline.py

import logging
import numpy as np
from typing import List, Tuple

import config
from .feature_extraction import FeatureExtractor
from .odometry_calculation import OdometryCalculator
from .frame_processor import FrameProcessor
from .trajectory_writer import TrajectoryWriter
from .imu_synchronizer import IMUSynchronizer
from .utils import quat_to_rotmat, normalize_quat, quat_mul
from scipy.spatial.transform import Rotation as R_s
from .kalman_filter import VIOFilter


class OdometryPipeline:
    """
    Гибридный VIO: дискретная IMU-предынтеграция + коррекция по Visual Odometry.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        odometry_calculator: OdometryCalculator,
        frame_processor: FrameProcessor,
        imu_synchronizer: IMUSynchronizer,
        logger: logging.Logger,
        trajectory_writer: TrajectoryWriter,
        window_size: int = config.WINDOW_SIZE,
        T_BS: np.ndarray = config.T_BS
    ):
        self.fe     = feature_extractor
        self.odom   = odometry_calculator
        self.fp     = frame_processor
        self.imu    = imu_synchronizer
        self.logger = logger
        self.writer = trajectory_writer

        self.window_size = window_size

        # экструзия body→sensor и обратная
        self.T_BS = T_BS
        self.R_BS = self.T_BS[:3, :3]
        self.R_SB = self.R_BS.T

        # глобальная ориентация и позиция
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3,1))

        dt = 1.0 / config.VO_FPS
        self.filter = VIOFilter(
            dt=dt, window_size=window_size,
            accel_noise=config.IMU_ACCEL_NOISE,
            vo_noise=config.VO_NOISE)

        self.logger.info("OdometryPipeline ready: IMU предынтеграция + VO")


    def run(self, loader):
        # 1) читаем первый кадр
        first = loader.read_next()
        if first is None:
            self.logger.error("Empty data source")
            return
        frame, prev_ts = first

        # 2) калибровка смещений IMU
        if self.imu is not None:
            self.imu.calibrate_bias(static_duration=2.0)
            self.logger.info(
                f"Calibrated IMU biases: gyro={self.imu.gyro_bias}, accel={self.imu.accel_bias}"
            )

        # 3) выравнивание начальной ориентации по гравитации
        if self.imu is not None:
            g_meas = (self.imu.accel_bias + np.array([0.,0.,-9.81]))
            g_meas /= np.linalg.norm(g_meas)
            v1, v2 = g_meas, np.array([0.,0.,-1.])
            axis = np.cross(v1, v2)
            if np.linalg.norm(axis) < 1e-6:
                self.R_total = np.eye(3)
            else:
                axis /= np.linalg.norm(axis)
                angle = np.arccos(np.clip(v1.dot(v2), -1.0, 1.0))
                self.R_total = R_s.from_rotvec(axis * angle).as_matrix()
            self.logger.info(f"Initial orientation aligned to gravity:\n{self.R_total}")

        # 4) пишем начальную позу
        self.writer.write_pose(prev_ts, self.t_total, self.R_total)

        # 5) инициализируем предынтегратор
        rot0    = R_s.from_matrix(self.R_total)
        q_scipy = rot0.as_quat()  # формат [x,y,z,w]
        # переведём в [w,x,y,z]
        q_total = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=float)
        v_total = np.zeros(3)
        p_total = self.t_total.flatten()

        idx = 1
        while True:
            item = loader.read_next()
            if item is None:
                self.logger.info("Pipeline finished.")
                break
            frame, ts = item
            idx += 1

            # 6) IMU-предынтеграция на отрезке prev_ts→ts
            if self.imu is not None:
                q_total, v_total, p_total = self.imu.preintegrate(
                    prev_ts, ts,
                    q0=q_total,
                    v0=v_total,
                    p0=p_total
                )
                self.R_total = quat_to_rotmat(q_total)
                self.t_total = p_total.reshape(3,1)

                _, _, accels = self.imu.get_window(prev_ts, ts)
                if len(accels) > 0:
                    accel_mean = np.mean(accels, axis=0)
                    self.filter.predict(accel_mean)

            # 7) Visual Odometry — коррекция от FrameProcessor
            res = self.fp.process_frame(frame, prev_ts, ts)
            if res is None:
                self.logger.warning(f"Frame {idx}: VO lost, pure IMU predict")
            else:
                _, _, (R_cam, t_cam) = res

                z = []
                for i in range(self.window_size):
                    z.append(t_cam.flatten())
                z = np.concatenate(z)    
                self.filter.update(z)

                # переводим из камеры в тело
                R_body = self.R_SB @ R_cam @ self.R_BS
                t_body = self.R_SB @ t_cam

                # корректируем глобальную позу
                self.R_total = R_body @ self.R_total
                self.t_total = self.t_total + self.R_total @ t_body

                # синхронизируем интегратор
                rot_corr = R_s.from_matrix(self.R_total)
                qc = rot_corr.as_quat()
                q_total = np.array([qc[3], qc[0], qc[1], qc[2]], dtype=float)
                p_total = self.t_total.flatten()
                v_total = R_body @ v_total

            # 8) записываем результат
            self.logger.info(f"Frame {idx}: position = {self.t_total.flatten()}")
            self.writer.write_pose(ts, self.t_total, self.R_total)
            prev_ts = ts

        self.writer.close()
