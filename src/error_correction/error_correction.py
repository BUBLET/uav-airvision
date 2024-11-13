import numpy as np
from typing import Tuple
import logging
from .kalman_filter import KalmanFilter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorCorrector:
    def __init__(self, dt: float, process_noise: float, measurement_noise: float):
        """
        Инициализация объекта для коррекции ошибок.

        Параметры:
        - dt (float): шаг времени между измерениями.
        - process_noise (float): уровень шума процесса.
        - measurement_noise (float): уровень шума измерений.
        """
        # Инициализируем фильтр Калмана с заданными параметрами
        self.kalman_filter = KalmanFilter(dt, process_noise, measurement_noise)
        logger.info("ErrorCorrector инициализирован с фильтром Калмана.")

    def apply_correction(self, raw_position: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет фильтр Калмана для коррекции ошибки в положении.

        Параметры:
        - raw_position (Tuple[float, float]): позиция в формате (x, y) из одометрии.

        Возвращает:
        - corrected_position (numpy.ndarray): откорректированная позиция (x, y).
        - corrected_velocity (numpy.ndarray): откорректированная скорость (vx, vy).

        Исключения:
        - ValueError: Если входные данные некорректны.
        """
        if raw_position is None or len(raw_position) != 2:
            raise ValueError("raw_position должен быть кортежем из двух элементов (x, y).")
        
        # Шаг предсказания в фильтре Калмана
        self.kalman_filter.predict()
        logger.debug("Шаг предсказания выполнен.")

        # Обновляем фильтр Калмана на основе нового измерения позиции
        measurement = np.array(raw_position).reshape(-1, 1)
        self.kalman_filter.update(measurement)
        logger.debug("Шаг обновления выполнен с измерением: {}".format(raw_position))

        # Получаем откорректированные позицию и скорость
        corrected_position = self.kalman_filter.get_position()
        corrected_velocity = self.kalman_filter.get_velocity()
        logger.debug("Откорректированная позиция: {}, скорость: {}".format(corrected_position, corrected_velocity))

        return corrected_position, corrected_velocity

    def predict_next_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполняет предсказание следующего состояния с помощью фильтра Калмана.

        Возвращает:
        - predicted_position (numpy.ndarray): предсказанная позиция (x, y).
        - predicted_velocity (numpy.ndarray): предсказанная скорость (vx, vy).
        """
        # Шаг предсказания в фильтре Калмана
        self.kalman_filter.predict()
        logger.debug("Шаг предсказания выполнен.")

        # Получаем предсказанные позицию и скорость
        predicted_position = self.kalman_filter.get_position()
        predicted_velocity = self.kalman_filter.get_velocity()
        logger.debug("Предсказанная позиция: {}, скорость: {}".format(predicted_position, predicted_velocity))

        return predicted_position, predicted_velocity
