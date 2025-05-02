import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
import config

class OdometryCalculator:
    def __init__(self, image_width: int, image_height: int,
                 camera_matrix: Optional[np.ndarray] = None,
                 logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.dist_coeffs   = config.DIST_COEFFS
        self.E_RANSAC_THRESHOLD = config.E_RANSAC_THRESHOLD
        if camera_matrix is None:
            self.camera_matrix = config.CAMERA_MATRIX
        else:
            self.camera_matrix = camera_matrix
        # Дисторсия из config.py
        self.dist_coeffs = config.DIST_COEFFS
        self.logger.info("Используются CAMERA_MATRIX и DIST_COEFFS из config")
        self.logger.info(f"  K =\n{self.camera_matrix}")
        self.logger.info(f"  dist = {self.dist_coeffs.ravel()}")

    def _undistort(self, pts: np.ndarray) -> np.ndarray:
        """
        Приводит Nx2 массив точек к undistorted Nx2,
        затем возвращает их уже в однородные координаты P=K*X
        """
        # OpenCV ожидает форму (N,1,2)
        pts_reshaped = pts.reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts_reshaped, self.K, self.dist, P=self.K)
        return und.reshape(-1, 2)

    def calculate_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray):
            # pts1, pts2 — (N,2) координаты в пикселях
            # 1) снимаем дисторсию:
            p1_und = cv2.undistortPoints(pts1.reshape(-1,1,2),
                                        self.camera_matrix, self.dist_coeffs)
            p2_und = cv2.undistortPoints(pts2.reshape(-1,1,2),
                                        self.camera_matrix, self.dist_coeffs)
            # 2) считаем Essential
            E, mask = cv2.findEssentialMat(
                p1_und, p2_und, cameraMatrix=np.eye(3),
                method=cv2.RANSAC, prob=0.999,
                threshold=self.E_RANSAC_THRESHOLD
            )
            if E is None:
                self.logger.warning("Не удалось вычислить Essential")
                return None
            inliers = int(mask.sum())
            self.logger.info(f"[E] inliers = {inliers}/{len(mask.ravel())}")
            return E, mask, 0.0

    def _recover_pose(
        self,
        E: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Возвращает R, t и число inliers. 
        """
        # предварительно исправляем искажения
        p1_und = cv2.undistortPoints(pts1.reshape(-1, 1, 2),
                                     self.camera_matrix, self.dist_coeffs)
        p2_und = cv2.undistortPoints(pts2.reshape(-1, 1, 2),
                                     self.camera_matrix, self.dist_coeffs)

        # recoverPose на уже декалиброванных точках, поэтому K = I
        retval, R, t, mask_pose = cv2.recoverPose(
            E, p1_und, p2_und, cameraMatrix=np.eye(3)
        )
        inliers = int((mask_pose > 0).sum())
        if inliers < config.MIN_INLIERS:
            self.logger.warning(f"recoverPose inliers < {config.MIN_INLIERS}")
            return None, None, 0

        self.logger.info(f"E→R,t: inliers={inliers}")
        return R, t, inliers

    def decompose_essential(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Сначала находим E, затем восстанавливаем R, t и число inliers.
        """
        res = self.calculate_essential_matrix(pts1, pts2)
        if res is None:
            return None, None, 0
        E, _, _ = res
        R_mat, t_vec, inliers = self._recover_pose(E, pts1, pts2)
        return R_mat, t_vec, inliers
