# tests/test_odometry_calculation.py

import unittest
from unittest.mock import MagicMock, patch
import itertools
import numpy as np
import cv2

from src.image_processing.odometry_calculation import OdometryCalculator
from src.image_processing.map_point import MapPoint

class TestOdometryCalculator(unittest.TestCase):
    def setUp(self):
        MapPoint._id_generator = itertools.count(0)
        self.camera_matrix = np.array([[800, 0, 320],
                                       [0, 800, 240],
                                       [0, 0, 1]], dtype=np.float64)
        self.calculator = OdometryCalculator(
            image_width=640,
            image_height=480,
            camera_matrix=self.camera_matrix,
            logger=MagicMock()
        )
        self.prev_keypoints = [
            cv2.KeyPoint(0, 0, 1),
            cv2.KeyPoint(10, 0, 1),
            cv2.KeyPoint(0, 10, 1),
            cv2.KeyPoint(10, 10, 1),
            cv2.KeyPoint(5, 5, 1),
            cv2.KeyPoint(15, 5, 1),
            cv2.KeyPoint(5, 15, 1),
            cv2.KeyPoint(15, 15, 1)
        ]
        self.curr_keypoints = [
            cv2.KeyPoint(5, 5, 1),
            cv2.KeyPoint(15, 0, 1),
            cv2.KeyPoint(5, 15, 1),
            cv2.KeyPoint(15, 15, 1),
            cv2.KeyPoint(10, 10, 1),
            cv2.KeyPoint(20, 5, 1),
            cv2.KeyPoint(10, 20, 1),
            cv2.KeyPoint(20, 20, 1)
        ]
        self.prev_descriptors = np.random.randint(0, 256, (8, 32), dtype=np.uint8)
        self.curr_descriptors = np.random.randint(0, 256, (8, 32), dtype=np.uint8)
        self.homography_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.5) for i in range(8)]
        self.triangulation_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.5) for i in range(8)]

    def test_calculate_homography_matrix_success(self):
        matches = self.homography_matches[:4]
        with patch.object(self.calculator, '_calculate_symmetric_transfer_error_homography', return_value=0.0):
            result = self.calculator.calculate_homography_matrix(
                self.prev_keypoints,
                self.curr_keypoints,
                matches
            )
        self.assertIsNotNone(result, "Homography calculation returned None")
        H, mask, error = result
        self.assertEqual(H.shape, (3, 3), "Homography matrix shape is incorrect")
        self.assertEqual(mask.shape, (4, 1), "Mask shape is incorrect")
        self.assertIsInstance(error, float, "Error is not a float")

    def test_get_inliers_epipolar_success(self):
        with patch('cv2.findFundamentalMat', return_value=(np.eye(3), np.ones((len(self.homography_matches[:8]), 1), dtype=np.uint8))):
            result = self.calculator.get_inliers_epipolar(
                self.prev_keypoints,
                self.curr_keypoints,
                self.homography_matches[:8]
            )
            self.assertEqual(len(result), 8, "Number of inliers is incorrect")

    def test_triangulate_new_map_points_success(self):
        keyframe1 = (0, self.prev_keypoints, self.prev_descriptors, np.hstack((np.eye(3), np.zeros((3,1)))))
        keyframe2 = (1, self.curr_keypoints, self.curr_descriptors, np.hstack((np.eye(3), np.array([[0], [0], [1]]))))
        matches = self.triangulation_matches[:8]
        with patch('cv2.triangulatePoints') as mock_triangulate:
            pts4D = np.vstack((
                np.random.rand(3, 8) * 10,
                np.ones((1, 8))
            ))
            mock_triangulate.return_value = pts4D
            new_map_points = self.calculator.triangulate_new_map_points(
                keyframe1,
                keyframe2,
                matches
            )
        self.assertEqual(len(new_map_points), 8, "Number of triangulated MapPoints is incorrect")
        for i, mp in enumerate(new_map_points):
            self.assertEqual(mp.id, i, f"MapPoint.id for point {i} is incorrect")
            self.assertEqual(mp.matched_times, 0, "MapPoint.matched_times is incorrect")
            self.assertEqual(len(mp.descriptors), 2, "Number of descriptors in MapPoint is incorrect")
            self.assertEqual(len(mp.observations), 2, "Number of observations in MapPoint is incorrect")

    def test_visible_map_points_success(self):
        mp1 = MapPoint(coordinates=np.array([0, 0, 5]))
        mp2 = MapPoint(coordinates=np.array([100, 100, 5]))
        mp3 = MapPoint(coordinates=np.array([50, 50, 10]))
        map_points = [mp1, mp2, mp3]
        curr_pose = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]], dtype=np.float64)
        with patch.object(self.calculator, '_match_projected_points_with_keypoints', return_value=([mp1, mp2], [0,1])):
            matched_map_points, matched_indices = self.calculator.visible_map_points(
                map_points,
                self.curr_keypoints,
                self.curr_descriptors,
                curr_pose
            )
            self.assertEqual(len(matched_map_points), 2, "Number of matched map points is incorrect")
            self.assertEqual(len(matched_indices), 2, "Number of matched indices is incorrect")

    def test_convert_points_to_structure_success(self):
        pts3D = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0]
        ], dtype=np.float32)
        inlier_matches = self.triangulation_matches[:8]
        result = self.calculator.convert_points_to_structure(
            pts3D,
            self.prev_keypoints,
            self.curr_keypoints,
            inlier_matches,
            self.prev_descriptors,
            self.curr_descriptors,
            prev_frame_idx=0,
            curr_frame_idx=1
        )
        self.assertEqual(len(result), 8, "Number of MapPoints created is incorrect")
        for i, mp in enumerate(result):
            self.assertEqual(mp.id, i, f"MapPoint.id for point {i} is incorrect")
            self.assertEqual(mp.matched_times, 0, "MapPoint.matched_times is incorrect")
            self.assertEqual(len(mp.descriptors), 2, "Number of descriptors in MapPoint is incorrect")
            self.assertEqual(len(mp.observations), 2, "Number of observations in MapPoint is incorrect")

    def test_filter_map_points_with_descriptors(self):
        mp1 = MapPoint(coordinates=np.array([0.5, 0.5, 1.0]))
        mp1.descriptors = [self.prev_descriptors[0]]
        mp2 = MapPoint(coordinates=np.array([1.0, 1.0, 2.0]))
        mp2.descriptors = []
        mp3 = MapPoint(coordinates=np.array([1.5, 1.5, 3.0]))
        mp3.descriptors = [self.prev_descriptors[1]]
        map_points = [mp1, mp2, mp3]
        filtered = self.calculator._filter_map_points_with_descriptors(map_points)
        self.assertEqual(len(filtered), 2, "Filtered map points count is incorrect")
        self.assertEqual(filtered[0].id, 0, "First filtered MapPoint.id is incorrect")
        self.assertEqual(filtered[1].id, 2, "Second filtered MapPoint.id is incorrect")

    def test_update_connections_after_pnp(self):
        mp1 = MapPoint(coordinates=np.array([0.5, 0.5, 1.0]))
        mp1.descriptors = [self.prev_descriptors[0], self.curr_descriptors[0]]
        map_points = [mp1]
        curr_keypoints = [cv2.KeyPoint(5.0, 5.0, 1)]
        curr_descriptors = np.array([self.curr_descriptors[0]], dtype=np.uint8)
        with patch.object(self.calculator, '_knn_match', return_value=[[cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.5)]]):
            with patch.object(self.calculator, '_lowe_ratio_test', return_value=[cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.5)]):
                self.calculator.update_connections_after_pnp(
                    map_points,
                    curr_keypoints,
                    curr_descriptors,
                    frame_idx=1
                )
        self.assertEqual(len(map_points[0].observations), 1, "MapPoint.observations count is incorrect")
        self.assertEqual(map_points[0].matched_times, 1, "MapPoint.matched_times was not updated correctly")
        self.assertEqual(map_points[0].observations[0], (1, 0), "Last observation is incorrect")

if __name__ == '__main__':
    unittest.main()
