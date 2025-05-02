import pandas as pd
import numpy as np
from typing import List, Tuple

class IMUSynchronizer:
    """
    Буферизует IMU-измерения и выдаёт сглаженные accel/gyro
    на произвольный момент frame_time (секунды).
    """
    def __init__(self, imu_csv_path: str):
        # читаем колонки: ts [ns], wx, wy, wz, ax, ay, az
        df = pd.read_csv(
            imu_csv_path,
            sep=',',
            skiprows=1,
            header=None,
            names=["ts","wx","wy","wz","ax","ay","az"],
            dtype={"ts": np.int64, "wx":float,"wy":float,"wz":float,
                   "ax":float,"ay":float,"az":float}
        )
        # переводим наносекунды → секунды
        self.times = df["ts"].values.astype(np.float64) * 1e-9
        self.gyro  = df[["wx","wy","wz"]].values
        self.accel = df[["ax","ay","az"]].values

    def get_measurements_for_frame(self, frame_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (gyro_interp, accel_interp) в момент frame_time.
        Делает одномерную линейную интерполяцию между ближайшими замерами.
        """
        # если frame_time вне диапазона, задвигаем в границы
        if frame_time <= self.times[0]:
            idx = 0
            return self.gyro[0], self.accel[0]
        if frame_time >= self.times[-1]:
            idx = -1
            return self.gyro[-1], self.accel[-1]

        # найдём индексы слева и справа
        right = np.searchsorted(self.times, frame_time)
        left = right - 1
        t0, t1 = self.times[left], self.times[right]
        w0, w1 = self.gyro[left],  self.gyro[right]
        a0, a1 = self.accel[left], self.accel[right]

        α = (frame_time - t0) / (t1 - t0)
        gyro_interp  = w0 + α * (w1 - w0)
        accel_interp = a0 + α * (a1 - a0)
        return gyro_interp, accel_interp
