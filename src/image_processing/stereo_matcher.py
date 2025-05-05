import numpy as np
import cv2

from .utils import skew

class StereoMatcher:
    def __init__(self,
                 lk_params,
                 imu_processor,
                 pyramid_builder,
                 camera_model,
                 stereo_threshold):
        """
        lk_params:         dict для cv2.calcOpticalFlowPyrLK
        imu_processor:     IMUProcessor с полями R_cam0_imu, R_cam1_imu, t_cam0_imu, t_cam1_imu и методом integrate_imu_data
        pyramid_builder:   PyramidBuilder с атрибутами curr_cam0_pyramid, curr_cam1_pyramid
        camera_model:      CameraModel с методами undistort_points и distort_points
        stereo_threshold:  порог для disparity и эпиполярного фильтра
        """
        self.lk_params       = lk_params
        self.integrate_imu   = imu_processor.integrate_imu_data
        self.R_cam0_imu      = imu_processor.R_cam0_imu
        self.R_cam1_imu      = imu_processor.R_cam1_imu
        self.t_cam0_imu      = imu_processor.t_cam0_imu
        self.t_cam1_imu      = imu_processor.t_cam1_imu

        self.pyr0            = pyramid_builder.curr_cam0_pyramid
        self.pyr1            = pyramid_builder.curr_cam1_pyramid

        self.camera_model    = camera_model
        self.stereo_threshold= stereo_threshold

    def stereo_match(self, cam0_points):
        """
        Matches points from cam0 to cam1 using optical flow + stereo geometry.

        Args:
            cam0_points: list or array of (x,y) in left image.

        Returns:
            cam1_points: ndarray of matched (x,y) in right image.
            inlier_mask: boolean ndarray, True for valid matches.
        """
        if len(cam0_points) == 0:
            return np.array([]), np.array([], dtype=bool)

        pts0 = np.array(cam0_points, dtype=np.float32)
        # rectify undistort
        R0to1 = self.R_cam1_imu.T @ self.R_cam0_imu
        und0 = self.camera_model.undistort_points(
            pts0, self.camera_model.intrinsics,
            self.camera_model.distortion_model,
            self.camera_model.distortion_coeffs,
            rectification_matrix=R0to1
        )
        # project back into cam1 distorted
        proj1 = self.camera_model.distort_points(
            und0, self.camera_model.intrinsics,
            self.camera_model.distortion_model,
            self.camera_model.distortion_coeffs
        )

        # LK-tracking left->right
        p1, track_mask, _ = cv2.calcOpticalFlowPyrLK(
            self.pyr0, self.pyr1,
            pts0, np.array(proj1, dtype=np.float32),
            **self.lk_params
        )
        # reverse check right->left
        p0r, rev_mask, _ = cv2.calcOpticalFlowPyrLK(
            self.pyr1, self.pyr0,
            p1, pts0.copy(),
            **self.lk_params
        )
        err = np.linalg.norm(pts0 - p0r, axis=1)
        disp = np.abs(proj1[:,1] - p1[:,1])

        inlier = (track_mask.reshape(-1).astype(bool) &
                  (err < 3) &
                  (disp < 20))

        # boundary check in right image
        h, w = self.pyr1.shape[:2]
        for i, pt in enumerate(p1):
            if not inlier[i]:
                continue
            x,y = pt
            if x<0 or x>=w or y<0 or y>=h:
                inlier[i] = False

        # essential-matrix filtering
        t01 = self.R_cam1_imu.T @ (self.t_cam0_imu - self.t_cam1_imu)
        E = skew(t01) @ R0to1

        undist0 = self.camera_model.undistort_points(
            pts0, self.camera_model.intrinsics,
            self.camera_model.distortion_model,
            self.camera_model.distortion_coeffs
        )
        undist1 = self.camera_model.undistort_points(
            p1, self.camera_model.intrinsics,
            self.camera_model.distortion_model,
            self.camera_model.distortion_coeffs
        )
        norm_unit = 4.0 / (2 * self.camera_model.intrinsics[0] +
                           2 * self.camera_model.intrinsics[1])
        for i, (u0, u1) in enumerate(zip(undist0, undist1)):
            if not inlier[i]:
                continue
            pt0_h = np.array([u0[0], u0[1], 1.0])
            pt1_h = np.array([u1[0], u1[1], 1.0])
            line = E @ pt0_h
            err_epi = abs((pt1_h * line)[0]) / np.linalg.norm(line[:2])
            if err_epi > self.stereo_threshold * norm_unit:
                inlier[i] = False

        return p1, inlier
