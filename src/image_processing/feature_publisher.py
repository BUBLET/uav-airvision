import numpy as np
import cv2
import time

from .feature_measurment import FeatureMeasurement

from itertools import chain, compress
from collections import defaultdict, namedtuple

class FeaturePublisher:
    def __init__(self,
                 cam0_intrinsics, cam0_dist_model, cam0_dist_coeffs,
                 cam1_intrinsics, cam1_dist_model, cam1_dist_coeffs):
        """
        те же параметры дисторсии и интринсики для двух камер
        """
        self.cam0_intrinsics = cam0_intrinsics
        self.cam0_dist_model = cam0_dist_model
        self.cam0_dist_coeffs = cam0_dist_coeffs
        self.cam1_intrinsics = cam1_intrinsics
        self.cam1_dist_model = cam1_dist_model
        self.cam1_dist_coeffs = cam1_dist_coeffs

    def undistort_points(self, pts_in, intrinsics, distortion_model, 
        distortion_coeffs, rectification_matrix=np.identity(3),
        new_intrinsics=np.array([1, 1, 0, 0])):
        """
        Arguments:
            pts_in: points to be undistorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.
            rectification_matrix:
            new_intrinsics:

        Returns:
            pts_out: undistorted points.
        """
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
        Arguments:
            pts_in: points to be distorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.

        Returns:
            pts_out: distorted points. (N, 2)
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
    
    def publish(self):
        """
        Publish the features on the current image including both the 
        tracked and newly detected ones.
        """
        curr_ids = []
        curr_cam0_points = []
        curr_cam1_points = []
        for feature in chain.from_iterable(self.curr_features):
            curr_ids.append(feature.id)
            curr_cam0_points.append(feature.cam0_point)
            curr_cam1_points.append(feature.cam1_point)

        curr_cam0_points_undistorted = self.undistort_points(
            curr_cam0_points, self.cam0_intrinsics,
            self.cam0_dist_model, self.cam0_dist_coeffs)
        curr_cam1_points_undistorted = self.undistort_points(
            curr_cam1_points, self.cam1_intrinsics,
            self.cam1_dist_model, self.cam1_dist_coeffs)

        features = []
        for i in range(len(curr_ids)):
            fm = FeatureMeasurement()
            fm.id = curr_ids[i]
            fm.u0 = curr_cam0_points_undistorted[i][0]
            fm.v0 = curr_cam0_points_undistorted[i][1]
            fm.u1 = curr_cam1_points_undistorted[i][0]
            fm.v1 = curr_cam1_points_undistorted[i][1]
            features.append(fm)

        feature_msg = namedtuple('feature_msg', ['timestamp', 'features'])(
            self.cam0_curr_img_msg.timestamp, features)
        return feature_msg
    
    def draw_features_stereo(self):
        img0 = self.cam0_curr_img_msg.image
        img1 = self.cam1_curr_img_msg.image

        kps0 = []
        kps1 = []
        matches = []
        for feature in chain.from_iterable(self.curr_features):
            matches.append(cv2.DMatch(len(kps0), len(kps0), 0))
            kps0.append(cv2.KeyPoint(*feature.cam0_point, 1))
            kps1.append(cv2.KeyPoint(*feature.cam1_point, 1))

        img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None, flags=2)
        cv2.imshow('stereo features', img)
        cv2.waitKey(1)