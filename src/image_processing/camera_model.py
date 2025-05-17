import numpy as np
import cv2

class CameraModel:
    def __init__(self, intrinsics, distortion_model, distortion_coeffs):
        """
        intrinsics: последовательность [fx, fy, cx, cy]
        distortion_model: строка, например 'radtan' или 'equidistant'
        distortion_coeffs: массив коэффициентов дисторсии
        """
        self.intrinsics = intrinsics
        self.distortion_model = distortion_model
        self.distortion_coeffs = distortion_coeffs

        # Build intrinsic matrix K
        fx, fy, cx, cy = intrinsics
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=float)


    def undistort_points(self, pts_in, intrinsics, distortion_model, 
        distortion_coeffs, rectification_matrix=np.identity(3),
        new_intrinsics=np.array([1, 1, 0, 0])):

        if len(pts_in) == 0:
            return []
        
        pts_in = np.reshape(pts_in, (-1, 1, 2))
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        K_new = np.array([
            [new_intrinsics[0], 0.0, new_intrinsics[2]],
            [0.0, new_intrinsics[1], new_intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            pts_out = cv2.fisheye.undistortPoints(pts_in, K, distortion_coeffs,
                rectification_matrix, K_new)
        else:   # default: 'radtan'
            pts_out = cv2.undistortPoints(pts_in, K, distortion_coeffs, None,
                rectification_matrix, K_new)
        return pts_out.reshape((-1, 2))

    def distort_points(self, pts_in, intrinsics, distortion_model, 
            distortion_coeffs):
        """
        Аргументы:
            pts_in: исходные точки для применения дисторсии.
            intrinsics: параметры внутренней калибровки камеры.
            distortion_model: модель дисторсии камеры.
            distortion_coeffs: коэффициенты дисторсии.

        Возвращает:
            pts_out: искажённые точки. (N, 2)
        """
        if len(pts_in) == 0:
            return []

        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            pts_out = cv2.fisheye.distortPoints(pts_in, K, distortion_coeffs)
        else:   # default: 'radtan'
            homogenous_pts = cv2.convertPointsToHomogeneous(pts_in)
            pts_out, _ = cv2.projectPoints(homogenous_pts, 
                np.zeros(3), np.zeros(3), K, distortion_coeffs)
        return pts_out.reshape((-1, 2))

    def predict_feature_tracking(self, input_pts, R_p_c, intrinsics):

        if len(input_pts) == 0:
            return []

        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        H = K @ R_p_c @ np.linalg.inv(K)

        compensated_pts = []
        for i in range(len(input_pts)):
            p1 = np.array([*input_pts[i], 1.0])
            p2 = H @ p1
            compensated_pts.append(p2[:2] / p2[2])
        return np.array(compensated_pts, dtype=np.float32)
    
    def rescale_points(self, pts1, pts2):

        scaling_factor = 0
        for pt1, pt2 in zip(pts1, pts2):
            scaling_factor += np.linalg.norm(pt1)
            scaling_factor += np.linalg.norm(pt2)

        scaling_factor = (len(pts1) + len(pts2)) / scaling_factor * np.sqrt(2)

        for i in range(len(pts1)):
            pts1[i] *= scaling_factor
            pts2[i] *= scaling_factor

        return pts1, pts2, scaling_factor
