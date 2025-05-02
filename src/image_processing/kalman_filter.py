# src/image_processing/kalman_filter.py

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple

class VIOFilter:
    """
    Линейный Калман-фильтр для объединения IMU (ускорение) и VO (смещение).
    Состояние x = [px,py,pz, vx,vy,vz]^T.
    Управление u = accel (ax,ay,az).
    Измерение z = скорость ≈ t_rel/dt.
    """
    def __init__(self, dt: float,
                 accel_noise: float = 0.1,
                 vo_noise: float = 0.05):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # Матрица перехода
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        self.kf.F = F

        # Управление через ускорение
        B = np.zeros((6,3))
        B[3,0] = B[4,1] = B[5,2] = dt
        self.kf.B = B

        # Наблюдаем только скорость
        H = np.zeros((3,6))
        H[0,3] = H[1,4] = H[2,5] = 1.0
        self.kf.H = H

        # Начальные ковариации
        self.kf.P *= 1e-3
        # Процессный шум (ускорение)
        Q = np.zeros((6,6)); Q[3:,3:] = np.eye(3) * accel_noise
        self.kf.Q = Q
        # Шум измерения (скорость из VO)
        self.kf.R = np.eye(3) * vo_noise

    def predict(self, accel: np.ndarray):
        """Шаг прогнозирования по accel (3,) в world frame."""
        self.kf.predict(u=accel)

    def update(self, t_rel: np.ndarray):
        """
        Коррекция по смещению t_rel (3,1): 
        измеряем скорость z = t_rel / dt
        """
        z = (t_rel.flatten() / self.dt)
        self.kf.update(z)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (pos(3,), vel(3,))."""
        x = self.kf.x
        return x[:3], x[3:]
