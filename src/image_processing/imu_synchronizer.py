import pandas as pd
import numpy as np
from typing import Tuple
from .utils import quat_mul, quat_to_rotmat, normalize_quat
import config

class IMUSynchronizer:
    """
    Buffers IMU measurements and
      - calibrate_bias() to estimate gyro/acc biases at startup
      - get_window() yields bias-subtracted IMU between two times
      - get_measurements_for_frame() yields bias-subtracted interp’d sample
    """
    def __init__(self, imu_csv_path: str):
        # читаем колонки: ts [ns], wx, wy, wz, ax, ay, az
        df = pd.read_csv(
            imu_csv_path,
            sep=',',
            skiprows=1,
            header=None,
            names=["ts","wx","wy","wz","ax","ay","az"],
            dtype={"ts": np.int64,
                   "wx": float, "wy": float, "wz": float,
                   "ax": float, "ay": float, "az": float}
        )
        # наносекунды → секунды
        self.times  = df["ts"].values.astype(np.float64) * 1e-9
        self.gyros  = df[["wx","wy","wz"]].values
        self.accels = df[["ax","ay","az"]].values

        # will be set in calibrate_bias()
        self.gyro_bias    = np.zeros(3)
        self.accel_bias   = np.zeros(3)
        self._calibrated  = False

    def calibrate_bias(self, static_duration: float = 2.0):
        """
        Estimate gyro_bias = mean(ω) and accel_bias = mean(a) – [0,0,−9.81]
        over the first static_duration seconds.
        Call this once at startup, before any predict/update.
        """
        t0 = self.times[0]
        idx_end = np.searchsorted(self.times, t0 + static_duration, side='right')
        # slice out the static interval
        gyro0  = self.gyros[:idx_end]
        accel0 = self.accels[:idx_end]
        # biases
        self.gyro_bias   = gyro0.mean(axis=0)
        # assume true gravity = [0,0,-9.81]
        self.accel_bias  = accel0.mean(axis=0) - np.array([0., 0., -9.81])
        self._calibrated = True

    def get_window(self, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns all IMU measurements between [t0, t1], with biases removed.
        """
        idx0 = np.searchsorted(self.times, t0, side='left')
        idx1 = np.searchsorted(self.times, t1, side='right')
        w = self.gyros[idx0:idx1]
        a = self.accels[idx0:idx1]
        if self._calibrated:
            w = w - self.gyro_bias[None,:]
            a = a - self.accel_bias[None,:]
        return self.times[idx0:idx1], w, a

    def get_measurements_for_frame(self, frame_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearly interpolate gyro/accel at exactly frame_time, bias-subtracted.
        """
        if frame_time <= self.times[0]:
            gyro, accel = self.gyros[0], self.accels[0]
        elif frame_time >= self.times[-1]:
            gyro, accel = self.gyros[-1], self.accels[-1]
        else:
            right = np.searchsorted(self.times, frame_time)
            left  = right - 1
            t0, t1 = self.times[left], self.times[right]
            w0, w1 = self.gyros[left],  self.gyros[right]
            a0, a1 = self.accels[left], self.accels[right]
            α = (frame_time - t0) / (t1 - t0)
            gyro  = w0 + α*(w1 - w0)
            accel = a0 + α*(a1 - a0)

        if self._calibrated:
            gyro  = gyro  - self.gyro_bias
            accel = accel - self.accel_bias
        return gyro, accel
    
    def preintegrate(self, t0: float, t1: float,
                     q0: np.ndarray = np.array([1.,0.,0.,0],dtype=np.float64),
                     v0: np.ndarray = np.zeros(3),
                     p0: np.ndarray = np.zeros(3)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple on‐the‐fly preintegration (ignores noise/cov).
        Returns (q, v, p) at t1 in world frame, given initial
        orientation q0 (w,x,y,z), velocity v0, position p0.
        """
        times, gyros, accels = self.get_window(t0, t1)
        q = q0.copy()
        v = v0.copy()
        p = p0.copy()

        for i in range(len(times)-1):
            dt = times[i+1] - times[i]
            ω = gyros[i]      # bias‐compensated
            α = accels[i]     # bias‐compensated

            # --- orientation update (quaternion) ---
            θ = np.linalg.norm(ω) * dt
            if θ > 1e-8:
                axis = ω / np.linalg.norm(ω)
                dq = np.concatenate(([np.cos(θ/2)], np.sin(θ/2)*axis))
                q = normalize_quat(quat_mul(q, dq))

            # --- rotation body→world ---
            Rwb = quat_to_rotmat(q)

            # --- velocity & position ---
            v = v + (Rwb @ α + config.GRAVITY) * dt
            p = p + v*dt + 0.5*(Rwb @ α + config.GRAVITY)*(dt**2)

        return q, v, p
