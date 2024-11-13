import numpy as np
from typing import Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, dt: float, process_noise: float, measurement_noise: float):
        """
        Инициализация фильтра Калмана.

        Параметры:
        - dt (float): шаг времени между измерениями.
        - process_noise (float): дисперсия шума процесса (квадрат стандартного отклонения).
        - measurement_noise (float): дисперсия шума измерений.
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Инициализация состояния: [x, y, vx, vy]
        self.state = np.zeros((4, 1))

        # Модель перехода состояния (F)
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Модель наблюдений (H)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Инициализация ковариационной матрицы состояния (P)
        position_variance = 1.0  # Предполагаемая дисперсия позиции
        velocity_variance = 1.0  # Предполагаемая дисперсия скорости
        self.P = np.diag([position_variance, position_variance, velocity_variance, velocity_variance])

        # Ковариационная матрица шума процесса (Q)
        q = self.process_noise
        dt = self.dt
        self.Q = q * np.array([[dt**4/4, 0, dt**3/2, 0],
                               [0, dt**4/4, 0, dt**3/2],
                               [dt**3/2, 0, dt**2, 0],
                               [0, dt**3/2, 0, dt**2]])

        # Ковариационная матрица шума измерений (R)
        self.R = np.eye(2) * self.measurement_noise

    def predict(self):
        """
        Шаг предсказания: прогнозирование состояния системы на следующий временной шаг.
        """
        # Обновляем состояние на основе модели перехода
        self.state = self.F @ self.state

        # Обновляем ковариацию ошибки состояния с учетом шума процесса
        self.P = self.F @ self.P @ self.F.T + self.Q
        logger.debug("Предсказанное состояние: {}".format(self.state.flatten()))
        logger.debug("Предсказанная ковариация: \n{}".format(self.P))

    def update(self, measurement: np.ndarray):
        """
        Шаг обновления: корректировка предсказанного состояния на основе нового измерения.

        Параметры:
        - measurement (numpy.ndarray): новое измерение позиции (x, y).
        """
        if measurement is None or measurement.shape[0] != 2:
            raise ValueError("Измерение должно быть массивом формы (2,) или (2, 1)")

        measurement = measurement.reshape(2, 1)

        # Вычисляем ошибку между измерением и предсказанным состоянием
        y = measurement - self.H @ self.state

        # Вычисляем ковариацию ошибки измерения
        S = self.H @ self.P @ self.H.T + self.R

        # Вычисляем коэффициент Калмана
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Обновляем состояние
        self.state = self.state + K @ y

        # Обновляем ковариацию
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P

        logger.debug("Обновлённое состояние: {}".format(self.state.flatten()))
        logger.debug("Обновлённая ковариация: \n{}".format(self.P))

    def get_position(self) -> np.ndarray:
        """
        Возвращает текущую позицию (x, y) из состояния.

        Возвращает:
        - numpy.ndarray: позиция в формате (x, y).
        """
        return self.state[0:2, 0]

    def get_velocity(self) -> np.ndarray:
        """
        Возвращает текущую скорость (vx, vy) из состояния.

        Возвращает:
        - numpy.ndarray: скорость в формате (vx, vy).
        """
        return self.state[2:4, 0]
