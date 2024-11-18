import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
from python_orb_slam3 import ORBExtractor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.extractor = ORBExtractor()
        logger.info("FeatureExtractor инициализирован с параметрами ORB.")

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Извлекает ключевые точки и дескрипторы из изображения с использованием ORB.

        Параметры:
        - image (numpy.ndarray): Входное изображение в формате BGR или градаций серого.

        Возвращает:
        - keypoints (list of cv2.KeyPoint): Список найденных ключевых точек.
        - descriptors (numpy.ndarray или None): Массив дескрипторов.

        Исключения:
        - ValueError: Если изображение некорректно.

        """
        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Изображение не может быть пустым и должно быть корректным numpy.ndarray.")

        if image.size == 0:
            raise ValueError("Изображение пустое.")

        # Преобразуем изображение в градации серого, если оно цветное
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Изображение преобразовано в градации серого.")
        elif len(image.shape) == 2:
            image_gray = image
        else:
            raise ValueError("Неподдерживаемый формат изображения.")

        # Извлекаем ключевые точки и дескрипторы
        keypoints, descriptors = self.extractor.detectAndCompute(image_gray)

        if descriptors is None:
            logger.warning("Дескрипторы не были найдены.")
            descriptors = []

        logger.info(f"Найдено {len(keypoints)} ключевых точек.")

        return keypoints, descriptors

    def draw_keypoints(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Отображает ключевые точки на изображении для визуализации.

        Параметры:
        - image (numpy.ndarray):
        - keypoints (list of cv2.KeyPoint): Список ключевых точек для отображения.

        Возвращает:
        - image_with_keypoints (numpy.ndarray): Изображение с нанесенными ключевыми точками.

        Исключения:
        - ValueError: Если изображение некорректно.

        """
        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Изображение не может быть пустым и должно быть корректным numpy.ndarray.")

        if image.size == 0:
            raise ValueError("Изображение пустое.")

        # Если изображение в градациях серого, преобразуем в BGR для отображения цветных ключевых точек
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image_with_keypoints = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DEFAULT
        )
        return image_with_keypoints
