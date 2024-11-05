import cv2
import numpy as np

class OdometryCalculator:
    def __init__(self, focal_length, principal_point):
        """
        Инициализация объекта OdometryCalculator.
        
        Параметры:
        - focal_length (float): фокусное расстояние камеры.
        - principal_point (tuple): координаты главной точки (cx, cy).
        """
        self.focal_length = focal_length
        self.principal_point = principal_point

    def calculate_motion(self, prev_keypoints, curr_keypoints, matches):
        """
        Вычисляет движение между предыдущими и текущими кадрами на основе ключевых точек.
        
        Параметры:
        - prev_keypoints (list): список ключевых точек предыдущего кадра.
        - curr_keypoints (list): список ключевых точек текущего кадра.
        - matches (list): список сопоставленных точек.

        Возвращает:
        - translation (numpy.ndarray): вектор смещения (dx, dy).
        - rotation (float): угол поворота в радианах.
        """
        # Определяем соответствующие точки
        src_points = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_points = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Вычисляем матрицу преобразования
        E, mask = cv2.findEssentialMat(src_points, dst_points, self.focal_length, self.principal_point)
        points, R, t, mask = cv2.recoverPose(E, src_points, dst_points)

        return t, R

