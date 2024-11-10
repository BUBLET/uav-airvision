import numpy as np
from .kalman_filter import KalmanFilter

class ErrorCorrector:
    def __init__(self, dt, process_noise, measurement_noise):
        """
        Инициализация объекта для коррекции ошибок.

        Параметры:
        - dt (float): шаг времени между измерениями.
        - process_noise (float): уровень шума процесса.
        - measurement_noise (float): уровень шума измерений.
        """
        # Инициализируем фильтр Калмана с заданными параметрами
        self.kalman_filter = KalmanFilter(dt, process_noise, measurement_noise)

    def apply_correction(self, raw_position):
        """
        Применяет фильтр Калмана для коррекции ошибки в положении.

        Параметры:
        - raw_position (tuple): позиция в формате (x, y) из одометрии.

        Возвращает:
        - corrected_position (numpy.ndarray): откорректированная позиция (x, y).
        - corrected_velocity (numpy.ndarray): откорректированная скорость (vx, vy).
        """
        # Обновляем фильтр Калмана на основе нового измерения позиции
        self.kalman_filter.update(np.array(raw_position[:2]).reshape(-1, 1))

        # Получаем откорректированные позицию и скорость
        corrected_position = self.kalman_filter.get_position()
        corrected_velocity = self.kalman_filter.get_velocity()

        return corrected_position, corrected_velocity

    def predict_next_state(self):
        """
        Выполняет предсказание следующего состояния с помощью фильтра Калмана.
        
        Возвращает:
        - predicted_position (numpy.ndarray): предсказанная позиция (x, y).
        - predicted_velocity (numpy.ndarray): предсказанная скорость (vx, vy).
        """
        # Шаг предсказания в фильтре Калмана
        self.kalman_filter.predict()

        # Получаем предсказанные позицию и скорость
        predicted_position = self.kalman_filter.get_position()
        predicted_velocity = self.kalman_filter.get_velocity()

        return predicted_position, predicted_velocity
