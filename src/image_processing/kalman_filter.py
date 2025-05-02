import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple

class VIOFilter:
    """
    Линейный Калман-фильтр для слияния IMU (accel) и VO (t_rel).
    Состояние x = [px,py,pz, vx,vy,vz]^T.
    Управление u = accel (ax,ay,az).
    Измерение z = скорость ≈ t_rel / dt.
    """
    def __init__(self, dt: float, accel_noise: float = 0.1, vo_noise: float = 0.05):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.set_dt(dt)

        # начальная ковариация
        self.kf.P *= 1e-3

        # процессный шум (по ускорению)
        Q = np.zeros((6, 6))
        Q[3:, 3:] = np.eye(3) * accel_noise
        self.kf.Q = Q

        # шум измерения (VO → скорость)
        self.kf.R = np.eye(3) * vo_noise

        # **ВАЖНО**: указываем, что мы измеряем лишь скорость (vx,vy,vz)
        H = np.zeros((3, 6))
        H[:, 3:] = np.eye(3)
        self.kf.H = H

    def set_dt(self, dt: float):
        """
        Обновить матрицы F и B при изменении шага времени.
        """
        # матрица перехода
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.kf.F = F

        # управление через ускорение
        B = np.zeros((6, 3))
        B[3, 0] = B[4, 1] = B[5, 2] = dt
        self.kf.B = B

        self.dt = dt

    def predict(self, accel: np.ndarray):
        """
        Predict-шаг по accel (3,) в world frame.
        """
        self.kf.predict(u=accel)

    def update(self, t_rel: np.ndarray):
        """
        Update-шаг по измерению t_rel (3,1):
        преобразуем в скорость z = t_rel / dt.
        """
        z = (t_rel.flatten() / self.dt)
        self.kf.update(z)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает:
          pos  – (3,1)-вектор позиции,
          vel  – (3,1)-вектор скорости.
        """
        x = self.kf.x
        # Приводим к одномерному массиву длиной 6
        x = np.asarray(x).flatten()
        pos = x[0:3].reshape(3, 1)
        vel = x[3:6].reshape(3, 1)
        return pos, vel

