import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdometryCalculator:
    def __init__(self, image_width: int, image_height: int, focal_length: Optional[float] = None):
        """
        Инициализация объекта OdometryCalculator.

        Параметры:
        - image_width (int): ширина изображения в пикселях.
        - image_height (int): высота изображения в пикселях.
        - focal_length (float, optional): фокусное расстояние камеры в пикселях.
        """
        self.camera_matrix = self.get_default_camera_matrix(image_width, image_height, focal_length)
        self.dist_coeffs = np.zeros((4, 1))  # Предполагаем отсутствие дисторсии
        logger.info("OdometryCalculator инициализирован с приблизительной матрицей камеры.")

    @staticmethod
    def get_default_camera_matrix(image_width: int, image_height: int, focal_length: Optional[float] = None) -> np.ndarray:
        """
        Создает приблизительную матрицу камеры на основе размеров изображения и фокусного расстояния.

        Параметры:
        - image_width (int): ширина изображения в пикселях.
        - image_height (int): высота изображения в пикселях.
        - focal_length (float, optional): фокусное расстояние в пикселях.

        Возвращает:
        - camera_matrix (numpy.ndarray): матрица внутренней калибровки камеры.
        """
        if focal_length is None:
            focal_length = 0.9 * max(image_width, image_height)  # Коэффициент можно настроить
        cx = image_width / 2
        cy = image_height / 2
        camera_matrix = np.array([[focal_length, 0, cx],
                                  [0, focal_length, cy],
                                  [0, 0, 1]], dtype=np.float64)
        return camera_matrix

    def calculate_motion(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Вычисляет движение между предыдущими и текущими кадрами на основе ключевых точек.

        Параметры:
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - R (numpy.ndarray): матрица вращения (3x3).
        - t (numpy.ndarray): вектор трансляции (3x1).
        - mask (numpy.ndarray): маска с информацией о надежных соответствиях.

        Если движение не может быть вычислено, возвращает None.
        """
        MIN_MATCH_COUNT = 8  # Минимальное количество соответствий

        if len(matches) < MIN_MATCH_COUNT:
            logger.warning(f"Недостаточно соответствий для одометрии: {len(matches)} найдено, {MIN_MATCH_COUNT} требуется.")
            return None

        # Определяем соответствующие точки
        src_points = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_points = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Вычисляем матрицу Essential
        E, mask = cv2.findEssentialMat(
            src_points,
            dst_points,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            logger.warning("Не удалось вычислить матрицу Essential.")
            return None

        # Восстанавливаем относительное положение камеры
        _, R, t, mask_pose = cv2.recoverPose(E, src_points, dst_points, self.camera_matrix)

        if R is None or t is None:
            logger.warning("Не удалось восстановить позу камеры.")
            return None

        logger.info("Успешно вычислено относительное движение между кадрами.")
        return R, t, mask_pose
