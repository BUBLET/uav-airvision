import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple

class VIOFilter:
    def __init__(self, dt: float, window_size: int, accel_noise: float = 0.1, vo_noise: float = 0.05):
        # Установим размер окна для фильтра
        self.window_size = window_size
        self.dt = dt
        self.kf = KalmanFilter(dim_x=6 * window_size, dim_z=3 * window_size)

        # Шум процесса
        Q = np.zeros((6 * window_size, 6 * window_size))
        for i in range(window_size):
            Q[3 * i: 3 * (i + 1), 3 * i: 3 * (i + 1)] = np.eye(3) * accel_noise
        self.kf.Q = Q

        # Шум измерений
        self.kf.R = np.eye(3 * window_size) * vo_noise

        # Хе матрица для скорости
        self.kf.H = np.zeros((3 * window_size, 6 * window_size))
        for i in range(window_size):
            self.kf.H[3 * i: 3 * (i + 1), 6 * i: 6 * (i + 1)] = np.eye(3, 6)
    
    def set_dt(self, dt: float):
        """
        Позволяет поменять dt после инициализации.
        """
        self.dt = dt

    def predict(self, accel: np.ndarray):
        """
        Прогнозирование нового состояния с учетом IMU.
        """
        self.kf.predict(u=accel)

    def update(self, t_rel: np.ndarray):
        """
        Обновление состояния с учетом измерений от визуальной одометрии.
        """
        # Мы предполагаем, что t_rel содержит смещения для каждого кадра в окне.
        # Мы просто приводим его к нужной размерности (3 * window_size, 1).
        
        # Преобразуем t_rel в одномерный массив (все изменения) и масштабируем на dt
        z = (t_rel.flatten() / self.dt)
        
        # Проверяем, что размерность соответствует 3 * window_size
        expected_size = 3 * self.window_size
        if z.shape[0] != expected_size:
            raise ValueError(f"Expected z to have shape ({expected_size},), but got {z.shape}")
        
        # Преобразуем в одномерный массив с нужной размерностью (3 * window_size, 1)
        z = z.reshape(-1, 1)
        
        # Обновляем фильтр с правильно подготовленным z
        self.kf.update(z)


    def get_state(self):
        """
        Получение текущего состояния системы.
        """
        return self.kf.x

