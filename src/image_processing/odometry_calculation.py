import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

class OdometryCalculator:
    def __init__(self, image_width: int, image_height: int,
                 camera_matrix: Optional[np.ndarray] = None,
                 e_ransac_threshold: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.image_width = image_width
        self.image_height = image_height
        self.E_RANSAC_THRESHOLD = e_ransac_threshold
        if camera_matrix is None:
            self.camera_matrix = np.array([[800, 0, image_width / 2],
                                           [0, 800, image_height / 2],
                                           [0, 0, 1]], dtype=np.float64)
            self.logger.info("Используется матрица камеры по умолчанию")
        else:
            self.camera_matrix = camera_matrix
            self.logger.info("Используется переданная матрица камеры")
        self.logger.info("OdometryCalculator инициализирован")

    def _extract_corresponding_points(self, prev_keypoints: List[cv2.KeyPoint],
                                        curr_keypoints: List[cv2.KeyPoint],
                                        pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Для простоты: ожидаем, что pts уже содержат координаты (например, от optical flow)
        # В простом случае мы не используем сопоставление по дескрипторам.
        # Если понадобится, можно добавить извлечение точек из списка keypoints.
        return pts, pts

    def calculate_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix,
                                        method=cv2.RANSAC, prob=0.999, threshold=self.E_RANSAC_THRESHOLD)
        if E is None:
            self.logger.warning("Не удалось вычислить Essential")
            return None
        inliers_count = int(mask.sum())
        error = 0.0  # Для простоты оставим ошибку нулевой
        self.logger.info(f"[E] Inliers = {inliers_count}")
        return E, mask, error


    def _recover_pose(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:

        retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        if retval < 1:
            self.logger.warning("recoverPose не смог восстановить достаточное число точек.")
            return None, None, None, 0
        mask_pose_flat = (mask_pose > 0).ravel()
        inliers_count = int(mask_pose_flat.sum())
        self.logger.info(f"E->R,t: inliers={inliers_count}/{len(mask_pose_flat)}")
        return R, t, mask_pose_flat, inliers_count

    def decompose_essential(self, prev_keypoints: List[cv2.KeyPoint], curr_keypoints: List[cv2.KeyPoint],
                             pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        # Здесь pts1 и pts2 – это координаты, полученные, например, через optical flow
        R, t, mask_pose_flat, inliers_count = self._recover_pose(self.calculate_essential_matrix(pts1, pts2)[0], pts1, pts2)
        return R, t, mask_pose_flat, inliers_count
