# src/image_processing/odometry_pipeline.py

import logging
import numpy as np
from typing import Tuple, List

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
    VO + IMU fusion with extrinsics T_BS applied.
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

        # extrinsics
        self.T_BS = T_BS
        self.R_BS = self.T_BS[:3,:3]    # body→sensor
        self.R_SB = self.R_BS.T         # sensor→body

        # sliding window
        self.window_poses     : List[Tuple[np.ndarray,np.ndarray]] = []
        self.window_relatives : List[Tuple[np.ndarray,np.ndarray]] = []

        # global pose
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3,1))

        # EKF
        self.dt = 1.0 / config.VO_FPS
        self.vio = VIOFilter(dt=self.dt,
                             accel_noise=config.IMU_ACCEL_NOISE,
                             vo_noise=config.VO_NOISE)
        self.logger.info(f"VIOFilter init dt={self.dt}")

    def pose_to_vector(self, Rmat, tvec):
        rot = R.from_matrix(Rmat).as_rotvec()
        return np.concatenate((rot, tvec.flatten()))

    def vector_to_pose(self, x):
        rot = x[:3]
        t   = x[3:].reshape(3,1)
        Rm  = R.from_rotvec(rot).as_matrix()
        return Rm, t

    def optimize_window(self):
        N = len(self.window_poses)
        if N < 2: return
        x0 = np.concatenate([self.pose_to_vector(*self.window_poses[i])
                             for i in range(1,N)])
        self.logger.info(f"BA window size={N}")
        res = least_squares(self._residual, x0, verbose=0,
                            xtol=1e-2, ftol=1e-2,
                            loss='huber', f_scale=1.0)
        # unpack...
        xopt = res.x
        new = [self.window_poses[0]]
        for i in range(1,N):
            xi = xopt[(i-1)*6:i*6]
            new.append(self.vector_to_pose(xi))
        self.window_poses = new
        self.R_total, self.t_total = self.window_poses[-1]

    def _residual(self, x):
        # omitted for brevity—unchanged from before
        return np.zeros(0)

    def run(self, loader):
        # --- 1) read first frame + timestamp ---
        first = loader.read_next()
        if first is None:
            self.logger.error("Empty data source")
            return
        frame, prev_ts = first

        # --- 2) IMU bias calibration over a short static period ---
        if self.imu is not None:
            # collect first 2s of IMU to estimate biases
            self.imu.calibrate_bias(static_duration=2.0)
            self.logger.info(
                f"Calibrated IMU biases: gyro={self.imu.gyro_bias}, accel={self.imu.accel_bias}"
            )

        # --- 3) Gravity alignment for initial R_total ---
        if self.imu is not None:
            # measured gravity direction in body frame
            g_meas = (self.imu.accel_bias + np.array([0., 0., -9.81]))
            g_meas /= np.linalg.norm(g_meas)
            # want to rotate that to [0,0,-1]
            v1 = g_meas
            v2 = np.array([0., 0., -1.])
            axis = np.cross(v1, v2)
            if np.linalg.norm(axis) < 1e-6:
                self.R_total = np.eye(3)
            else:
                axis  /= np.linalg.norm(axis)
                angle = np.arccos(np.clip(v1.dot(v2), -1.0, 1.0))
                self.R_total = R.from_rotvec(axis * angle).as_matrix()
            self.logger.info(f"Initial orientation aligned to gravity:\n{self.R_total}")

        # --- 4) write initial pose & push into sliding window ---
        self.writer.write_pose(prev_ts, self.t_total, self.R_total)
        self.window_poses.append((self.R_total.copy(), self.t_total.copy()))

        # now continue exactly as before:
        idx = 1
        while True:
            item = loader.read_next()
            if item is None:
                self.logger.info("done")
                break
            frame, ts = item
            idx += 1

            # IMU predict
            times, _, accels = self.imu.get_window(prev_ts, ts)
            for i in range(len(times)-1):
                dti = times[i+1] - times[i]
                self.vio.set_dt(dti)
                self.vio.predict(accels[i])

            # VO
            res = self.fp.process_frame(frame)
            if res is None:
                self.logger.warning(f"frame {idx} skipped")
                prev_ts = ts
                continue
            _, _, (R_cam, t_cam) = res

            # 1) transform cam→body
            R_body = self.R_SB @ R_cam @ self.R_BS
            t_body = self.R_SB @ t_cam

            # 2) EKF‐update w/ body‐frame Δ‐translation
            self.vio.update(t_body)

            # 3) read back
            pos, _ = self.vio.get_state()
            self.t_total = pos
            self.R_total = R_body @ self.R_total

            # 4) window
            self.window_relatives.append((R_body.copy(), t_body.copy()))
            self.window_poses.append((self.R_total.copy(), self.t_total.copy()))
            if len(self.window_poses) > self.window_size:
                self.optimize_window()
                self.window_poses     = [(self.R_total.copy(), self.t_total.copy())]
                self.window_relatives = []

            self.logger.info(f"Frame {idx}: t={self.t_total.ravel()}")
            self.writer.write_pose(ts, self.t_total, self.R_total)
            prev_ts = ts

        self.writer.close()
        self.logger.info("finished")
