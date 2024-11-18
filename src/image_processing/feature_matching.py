import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureMatcher:
    def __init__(
        self,
        norm_type: int = cv2.NORM_HAMMING,
        cross_check: bool = False,
        matcher_type: str = 'BF',
        knn_k: int = 2,
        ratio_threshold: float = 0.75
    ):
        """
        Инициализация объекта FeatureMatcher с использованием BFMatcher или FLANN.

        Параметры:
        - norm_type (int): Тип нормализации для BFMatcher (cv2.NORM_HAMMING или cv2.NORM_L2).
        - cross_check (bool): Включить или отключить взаимную проверку в BFMatcher.
        - matcher_type (str): Тип матчера ('BF' для Brute-Force или 'FLANN' для FLANN-базированного).
        - knn_k (int): Количество ближайших соседей для KNN-сопоставления.
        - ratio_threshold (float): Порог для теста соотношения Лоу.

        """
        self.matcher_type = matcher_type
        self.knn_k = knn_k
        self.ratio_threshold = ratio_threshold

        if self.matcher_type == 'BF':
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)
            logger.info("Используется BFMatcher.")
        elif self.matcher_type == 'FLANN':
            # Параметры для FLANN
            if norm_type == cv2.NORM_HAMMING:
                index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=1)
            else:
                index_params = dict(algorithm=1,  # FLANN_INDEX_KDTREE
                                    trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            logger.info("Используется FlannBasedMatcher.")
        else:
            raise ValueError("Неподдерживаемый тип матчера. Используйте 'BF' или 'FLANN'.")

    def match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Сопоставляет дескрипторы между двумя изображениями с использованием KNN-сопоставления и теста соотношения Лоу.

        Параметры:
        - descriptors1 (numpy.ndarray): Дескрипторы первого изображения.
        - descriptors2 (numpy.ndarray): Дескрипторы второго изображения.

        Возвращает:
        - good_matches (list of cv2.DMatch): Список отфильтрованных сопоставлений.

        Исключения:
        - ValueError: Если дескрипторы некорректны или пустые.

        """
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Дескрипторы не могут быть None.")

        if len(descriptors1) == 0 or len(descriptors2) == 0:
            logger.warning("Один из наборов дескрипторов пустой.")
            return []

        # Выполняем KNN-сопоставление
        try:
            if self.matcher_type == 'BF':
                matches = self.matcher.knnMatch(descriptors1, descriptors2, k=self.knn_k)
            elif self.matcher_type == 'FLANN':
                # FLANN требует, чтобы дескрипторы были типа float32
                if descriptors1.dtype != np.float32:
                    descriptors1 = np.float32(descriptors1)
                if descriptors2.dtype != np.float32:
                    descriptors2 = np.float32(descriptors2)
                matches = self.matcher.knnMatch(descriptors1, descriptors2, k=self.knn_k)
        except cv2.error as e:
            logger.error(f"Ошибка при сопоставлении: {e}")
            return []

        # Применяем тест соотношения Лоу
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        logger.info(f"Найдено {len(good_matches)} хороших соответствий.")

        return good_matches

    def draw_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches_to_draw: int = 50
    ) -> np.ndarray:
        """
        Отображает совпадения между двумя изображениями.

        Параметры:
        - img1 (numpy.ndarray): Первое изображение.
        - img2 (numpy.ndarray): Второе изображение.
        - keypoints1 (list of cv2.KeyPoint): Ключевые точки первого изображения.
        - keypoints2 (list of cv2.KeyPoint): Ключевые точки второго изображения.
        - matches (list of cv2.DMatch): Список сопоставленных точек.
        - max_matches_to_draw (int): Максимальное количество совпадений для отображения.

        Возвращает:
        - result_image (numpy.ndarray): Изображение с отображенными совпадениями.

        """
        if img1 is None or img2 is None:
            raise ValueError("Изображения не могут быть пустыми.")

        # Ограничиваем количество отображаемых совпадений
        matches_to_draw = matches[:max_matches_to_draw]

        result_image = cv2.drawMatches(
            img1,
            keypoints1,
            img2,
            keypoints2,
            matches_to_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return result_image