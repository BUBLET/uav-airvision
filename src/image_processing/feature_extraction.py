import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(
        self,
        n_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        WTA_K: int = 2,
        score_type: int = cv2.ORB_HARRIS_SCORE,
        patch_size: int = 31,
        fast_threshold: int = 20,
    ):
        """
        Инициализация объекта FeatureExtractor с использованием ORB.

        Параметры:
        - n_features (int): Максимальное число ключевых точек для извлечения.
        - scale_factor (float): Коэффициент масштаба между уровнями пирамиды.
        - n_levels (int): Количество уровней в пирамиде.
        - edge_threshold (int): Размер границы на изображении, где ключевые точки не будут искаться.
        - first_level (int): Первый уровень пирамиды.
        - WTA_K (int): Параметр для выбора количества точек сравнения (2, 3, 4).
        - score_type (int): Тип метода оценки (cv2.ORB_HARRIS_SCORE или cv2.ORB_FAST_SCORE).
        - patch_size (int): Размер патча, используемый при вычислении дескрипторов.
        - fast_threshold (int): Порог для детектора FAST.

        """
        self.extractor = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=WTA_K,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )
        logger.info("FeatureExtractor инициализирован с параметрами ORB.")

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Извлекает ключевые точки и дескрипторы из изображения с использованием ORB.

        Параметры:
        - image (numpy.ndarray): Входное изображение в формате BGR или градаций серого.

        Возвращает:
        - keypoints (list of cv2.KeyPoint): Список найденных ключевых точек.
        - descriptors (numpy.ndarray или None): Массив дескрипторов или None, если дескрипторы не найдены.

        Исключения:
        - ValueError: Если изображение некорректно.

        """
        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Изображение не может быть пустым и должно быть корректным numpy.ndarray.")

        if image.size == 0:
            raise ValueError("Изображение пустое.")

        # Проверяем, имеет ли изображение три канала (цветное)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Преобразуем изображение в градации серого
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Изображение преобразовано в градации серого.")
        elif len(image.shape) == 2:
            # Изображение уже в градациях серого
            pass
        else:
            raise ValueError("Неподдерживаемый формат изображения.")

        # Извлекаем ключевые точки и дескрипторы
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)

        if descriptors is None:
            logger.warning("Дескрипторы не были найдены.")
            descriptors = []

        logger.info(f"Найдено {len(keypoints)} ключевых точек.")

        return keypoints, descriptors

    def draw_keypoints(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Отображает ключевые точки на изображении для визуализации.

        Параметры:
        - image (numpy.ndarray): Входное изображение в формате BGR или градаций серого.
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

# Пример использования
if __name__ == "__main__":
    # Задаём путь к изображению
    image_path = "test_image.jpg"

    # Загружаем изображение
    image = cv2.imread(image_path)

    if image is None:
        logger.error(f"Не удалось загрузить изображение по пути: {image_path}")
        exit(1)

    # Создаем объект FeatureExtractor с пользовательскими параметрами (при необходимости)
    feature_extractor = FeatureExtractor(n_features=1000)

    # Извлекаем ключевые точки и дескрипторы
    keypoints, descriptors = feature_extractor.extract_features(image)

    # Отображаем ключевые точки на изображении
    image_with_keypoints = feature_extractor.draw_keypoints(image, keypoints)

    # Показ изображения с ключевыми точками
    cv2.imshow("Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
