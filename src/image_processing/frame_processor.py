import logging
import cv2
import numpy as np
import config
from typing import Optional, Tuple, List
from .utils import quat_to_rotmat, normalize_quat
from .imu_synchronizer import IMUSynchronizer

class FrameProcessor:
    def __init__(self, 
                 feature_extractor, 
                 odometry_calculator, 
                 lk_params: dict,
                 imu_synchronizer: IMUSynchronizer):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = feature_extractor
        self.odometry_calculator = odometry_calculator
        self.lk_params = lk_params
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_ts: Optional[float] = None
        self.imu = imu_synchronizer

    def process_frame(
        self, 
        current_frame: np.ndarray,
        ts_prev: float,
        ts_curr: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Returns (current_gray, tracked_pts, (R_cam, t_cam)), or None on failure.
        """
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 1) First frame: detect features
        if self.prev_gray is None:
            pts = cv2.goodFeaturesToTrack(
                current_gray,
                maxCorners=config.N_FEATURES,
                qualityLevel=config.SHI_QUALITY,
                minDistance=config.SHI_MIN_DIST,
                blockSize=config.SHI_BLOCK_SIZE
            )
            if pts is None:
                self.logger.warning("SHI-TOMASI no corners first frame")
                return None
            self.prev_gray = current_gray
            self.prev_pts = pts
            return current_gray, self.prev_pts, (np.eye(3), np.zeros((3,1)))
        # 2) Track via LK
        gyro0, _ = self.imu.get_measurements_for_frame(ts_prev)
        gyro1, _ = self.imu.get_measurements_for_frame(ts_curr)
        omega = 0.5 * (gyro0 + gyro1)
        dt = ts_curr - ts_prev
        theta = np.linalg.norm(omega) * dt
        if theta > 1e-6:
            axis = omega / np.linalg.norm(omega)
            dq = np.concatenate(([np.cos(theta/2)], np.sin(theta/2)*axis))
            R_pred = quat_to_rotmat(dq)
            h, w = self.prev_gray.shape
            cx, cy = w/2, h/2
            pts = self.prev_pts.reshape(-1,2)
            pts_c = pts - [cx, cy]
            pts_rot = (R_pred[:2,:2] @ pts_c.T).T + [cx, cy]
            init_guess = pts_rot.reshape(-1,1,2).astype(np.float32)
        else:
            init_guess = None

        pts_prev = self.prev_pts.astype(np.float32).copy()
        if init_guess is not None:
            init_guess = init_guess.astype(np.float32).copy()
        
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            current_gray,
            pts_prev,
            init_guess,
            **self.lk_params
        )

        status = status.ravel().astype(bool)
        prev_good = pts_prev[status]
        next_good = next_pts[status]
        
        # 3) Forward‐backward check
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, self.prev_gray, next_good, None, **self.lk_params
        )
        fb_error = np.linalg.norm(prev_good.reshape(-1,2) - back_pts.reshape(-1,2), axis=1)
        mask_fb = fb_error < config.FB_ERROR_THRESH
        prev_good = prev_good[mask_fb]
        next_good = next_good[mask_fb]

        # 4) Re‐detect if too few
        if len(prev_good) < config.MIN_TRACKED:
            self.logger.warning(f"Only {len(prev_good)} tracks < {config.MIN_TRACKED}, re‐detect via SHI-TOMASI.")
            pts = cv2.goodFeaturesToTrack(
                current_gray,
                maxCorners=config.N_FEATURES,
                qualityLevel=config.SHI_QUALITY,
                minDistance=config.SHI_MIN_DIST,
                blockSize=config.SHI_BLOCK_SIZE
            )
            if pts is None:
                self.logger.warning("Shi-Tomasi re-detection failed.")
                return None
            self.prev_pts = pts            
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, current_gray, self.prev_pts, None, **self.lk_params
            )
            status = status.ravel().astype(bool)
            prev_good = self.prev_pts[status]
            next_good = next_pts[status]

        # 5) Need at least MIN_INLIERS for Essential
        if len(prev_good) < config.MIN_INLIERS:
            self.logger.warning(f"Too few points ({len(prev_good)}<{config.MIN_INLIERS}). skip.")
            self.prev_gray = current_gray
            self.prev_pts  = next_good.reshape(-1,1,2)
            return None

        # 6) Compute camera‐frame Δ‐pose
        pts1 = next_good.reshape(-1,2)
        pts2 = prev_good.reshape(-1,2)

        F, maskF = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=config.RANSAC_THRESHOLD,
            confidence=0.99
        )
        if maskF is not None:
            m = maskF.ravel().astype(bool)
            pts1 = pts1[m]
            pts2 = pts2[m]
            
        R_cam, t_cam, inliers = self.odometry_calculator.decompose_essential(pts1, pts2)
        if R_cam is None or t_cam is None or inliers < config.MIN_INLIERS:
            self.logger.warning(f"recoverPose failed ({inliers}<{config.MIN_INLIERS}).")
            self.prev_gray = current_gray
            self.prev_pts  = next_good.reshape(-1,1,2)
            return None

        # 7) Success
        self.prev_gray = current_gray
        self.prev_pts  = next_good.reshape(-1,1,2)
        self.prev_ts = ts_curr
        return current_gray, self.prev_pts, (R_cam, t_cam)
