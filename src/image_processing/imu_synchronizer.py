import pandas as pd
import numpy as np
from typing import Tuple

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
