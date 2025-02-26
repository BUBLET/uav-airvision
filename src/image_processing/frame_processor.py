import logging
import numpy as np
import cv2
from typing import List, Optional, Tuple

class FrameProcessor:
    def __init__(self, feature_extractor, odometry_calculator, lk_params: dict):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = feature_extractor
        self.odometry_calculator = odometry_calculator
        self.lk_params = lk_params
        self.prev_gray = None
        self.prev_pts = None

    def process_frame(self, current_frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            # Первый кадр: детектируем точки
            keypoints, _ = self.feature_extractor.extract_features(current_frame)
            if not keypoints:
                self.logger.warning("Не найдены точки на первом кадре.")
                return None
            self.prev_gray = current_gray
            self.prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            # Возвращаем тождественную позу
            return current_gray, self.prev_pts, (np.eye(3), np.zeros((3, 1)))
        
        # Отслеживаем точки через optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None, **self.lk_params)
        status = status.reshape(-1)
        good_prev_pts = self.prev_pts[status == 1]
        good_next_pts = next_pts[status == 1]

        # Двусторонняя проверка
        prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(current_gray, self.prev_gray, good_next_pts, None, **self.lk_params)
        status_back = status_back.reshape(-1)

        fb_error = np.linalg.norm(good_prev_pts.reshape(-1, 2) - prev_pts_back.reshape(-1, 2), axis=1)
        # ПОРОГ ОШИБКИ (УТОЧНИТЬ)
        fb_thresh = 0.5

        fb_mask = fb_error < fb_thresh
        good_prev_pts = good_prev_pts[fb_mask]
        good_next_pts = good_next_pts[fb_mask]
        
        # Если точек недостаточно (<50), повторно детектируем особенности (как в исходном коде)
        if len(good_prev_pts) < 50:
            self.logger.warning("Мало отслеживаемых точек (<50). Проводится повторное обнаружение.")
            # Используем prev_gray вместо current_frame для обнаружения, как в исходном коде:
            keypoints, _ = self.feature_extractor.extract_features(self.prev_gray)
            if not keypoints:
                self.logger.warning("Детектирование точек не удалось.")
                return None
            self.prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None, **self.lk_params)
            status = status.reshape(-1)
            good_prev_pts = self.prev_pts[status == 1]
            good_next_pts = next_pts[status == 1]

        
        if len(good_prev_pts) < 3:
            self.logger.warning("Недостаточно точек для расчета позы.")
            keypoints, _ = self.feature_extractor.extract_features(current_frame)
            if not keypoints:
                self.logger.warning("Детектирование точек не удалось.")
                return None
            self.prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            self.prev_gray = current_gray
            return current_gray, self.prev_pts, (np.eye(3), np.zeros((3, 1)))
        
        E, mask, error = self.odometry_calculator.calculate_essential_matrix(good_next_pts.reshape(-1, 2),
                                                                            good_prev_pts.reshape(-1, 2))
        if E is None:
            self.logger.warning("Не удалось вычислить эссенциальную матрицу.")
            self.prev_gray = current_gray
            self.prev_pts = good_next_pts.reshape(-1, 1, 2)
            return None
        R, t, mask_pose_flat, inliers_count = self.odometry_calculator._recover_pose(E,
                                                                                    good_next_pts.reshape(-1, 2),
                                                                                    good_prev_pts.reshape(-1, 2))
        if R is None or t is None or inliers_count < 5:
            self.logger.warning("Не удалось восстановить позу.")
            self.prev_gray = current_gray
            self.prev_pts = good_next_pts.reshape(-1, 1, 2)
            return None
        
        # Обновляем предыдущие данные для следующей итерации
        self.prev_gray = current_gray
        self.prev_pts = good_next_pts.reshape(-1, 1, 2)
        return current_gray, self.prev_pts, (R, t)
