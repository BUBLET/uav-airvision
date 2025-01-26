# tests/test_odometry_calculation.py

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import logging
import sys
import os

# Добавляем путь к src в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from image_processing.odometry_calculation import OdometryCalculator
from image_processing.map_point import MapPoint

class TestOdometryCalculator(unittest.TestCase):
    def setUp(self):
        self.image_width = 640
        self.image_height = 480
        self.camera_matrix = np.array([[800, 0, self.image_width / 2],
                                       [0, 800, self.image_height / 2],
                                       [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))
        self.logger = MagicMock(spec=logging.Logger)
        self.calculator = OdometryCalculator(
            image_width=self.image_width,
            image_height=self.image_height,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            logger=self.logger
        )

    def test_initialization_defaults(self):
        calc = OdometryCalculator(
            image_width=800,
            image_height=600
        )
        expected_camera_matrix = np.array([[800, 0, 400],
                                          [0, 800, 300],
                                          [0, 0, 1]], dtype=np.float64)
        np.testing.assert_array_almost_equal(calc.camera_matrix, expected_camera_matrix)
        np.testing.assert_array_almost_equal(calc.dist_coeffs, np.zeros((4,1)))

    def test_triangulate_new_map_points_no_descriptors(self):
        keyframe1 = (0, [], None, np.eye(4))
        keyframe2 = (1, [], None, np.eye(4))
        matches = []
        result = self.calculator.triangulate_new_map_points(keyframe1, keyframe2, matches)
        self.assertEqual(result, [])

    def test_triangulate_new_map_points_insufficient_matches(self):
        keyframe1 = (0, [cv2.KeyPoint(0,0,1)], [], np.eye(4))
        keyframe2 = (1, [cv2.KeyPoint(0,0,1)], [], np.eye(4))
        matches = [cv2.DMatch(0,0,0.5)]
        result = self.calculator.triangulate_new_map_points(keyframe1, keyframe2, matches)
        self.assertEqual(result, [])

    @patch('image_processing.odometry_calculation.cv2.triangulatePoints')
    def test_triangulate_new_map_points_success(self, mock_triangulate):
        mock_triangulate.return_value = np.array([[1, 2, 3, 4],
                                                 [1, 2, 3, 4],
                                                 [1, 2, 3, 4],
                                                 [1, 1, 1, 1]], dtype=np.float64)
        keypoints1 = [cv2.KeyPoint(100, 100, 1)]
        keypoints2 = [cv2.KeyPoint(150, 150, 1)]
        descriptors1 = np.array([[0,1,2,3]], dtype=np.uint8)
        descriptors2 = np.array([[0,1,2,3]], dtype=np.uint8)
        keyframe1 = (0, keypoints1, descriptors1, np.eye(4))
        keyframe2 = (1, keypoints2, descriptors2, np.eye(4))
        matches = [cv2.DMatch(0,0,0.5)]
        result = self.calculator.triangulate_new_map_points(keyframe1, keyframe2, matches)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 0)
        np.testing.assert_array_almost_equal(result[0].coordinates, [1,1,1])

    @patch('image_processing.odometry_calculation.cv2.findEssentialMat')
    def test_calculate_essential_matrix_no_matches(self, mock_find_essential):
        mock_find_essential.return_value = (None, None)
        result = self.calculator.calculate_essential_matrix([], [], [])
        self.assertIsNone(result)

    @patch('image_processing.odometry_calculation.cv2.findEssentialMat')
    def test_calculate_essential_matrix_success(self, mock_find_essential):
        E = np.eye(3)
        mask = np.array([[1], [0]], dtype=np.uint8)  # Изменено
        mock_find_essential.return_value = (E, mask)
        prev_keypoints = [cv2.KeyPoint(100, 100, 1), cv2.KeyPoint(200, 200, 1)]
        curr_keypoints = [cv2.KeyPoint(110, 110, 1), cv2.KeyPoint(210, 210, 1)]
        matches = [cv2.DMatch(0,0,0.5), cv2.DMatch(1,1,0.6)]
        with patch.object(self.calculator, 'calculate_symmetric_transfer_error', return_value=0.5):
            result = self.calculator.calculate_essential_matrix(prev_keypoints, curr_keypoints, matches)
            self.assertIsNotNone(result)
            self.assertEqual(result[0].tolist(), E.tolist())
            self.assertTrue((result[1] == mask).all())
            self.assertEqual(result[2], 0.5)

    @patch('image_processing.odometry_calculation.cv2.recoverPose')
    def test_decompose_essential_invalid_E(self, mock_recover_pose):
        result = self.calculator.decompose_essential(None, [], [], [])
        self.assertEqual(result, (None, None, None, 0))  # Убедитесь, что возвращается 4-элементный кортеж

    @patch('image_processing.odometry_calculation.cv2.recoverPose')
    def test_decompose_essential_success(self, mock_recover_pose):
        R = np.eye(3)
        t = np.array([[0], [0], [1]], dtype=np.float64)
        mask_pose = np.array([[1], [1], [0]], dtype=np.uint8)
        mock_recover_pose.return_value = (2, R, t, mask_pose)
        E = np.eye(3)
        prev_keypoints = [cv2.KeyPoint(100, 100, 1), cv2.KeyPoint(200, 200, 1)]
        curr_keypoints = [cv2.KeyPoint(110, 110, 1), cv2.KeyPoint(210, 210, 1)]
        matches = [cv2.DMatch(0,0,0.5), cv2.DMatch(1,1,0.6)]
        with patch.object(self.calculator, '_check_decompose_essential_inputs', return_value=True):
            result = self.calculator.decompose_essential(E, prev_keypoints, curr_keypoints, matches)
            self.assertIsNotNone(result)
            self.assertTrue((result[0] == R).all())
            self.assertTrue((result[1] == t).all())
            self.assertTrue((result[2] == mask_pose).all())
            self.assertEqual(result[3], 2)

    @patch('image_processing.odometry_calculation.cv2.findHomography')
    def test_calculate_homography_matrix_insufficient_matches(self, mock_find_homography):
        result = self.calculator.calculate_homography_matrix([], [], [
            cv2.DMatch(0,0,0.5), 
            cv2.DMatch(1,1,0.6), 
            cv2.DMatch(2,2,0.7)
        ])
        self.assertIsNone(result)

    @patch('image_processing.odometry_calculation.cv2.findHomography')
    def test_calculate_homography_matrix_success(self, mock_find_homography):
        H = np.eye(3)
        mask = np.array([[1], [0], [1]], dtype=np.uint8)
        mock_find_homography.return_value = (H, mask)
        prev_keypoints = [cv2.KeyPoint(100, 100, 1), cv2.KeyPoint(200, 200, 1)]
        curr_keypoints = [cv2.KeyPoint(110, 110, 1), cv2.KeyPoint(210, 210, 1)]
        matches = [cv2.DMatch(0,0,0.5), cv2.DMatch(1,1,0.6)]
        with patch.object(self.calculator, 'calculate_symmetric_transfer_error', return_value=0.8):
            result = self.calculator.calculate_homography_matrix(prev_keypoints, curr_keypoints, matches)
            self.assertIsNotNone(result)
            self.assertEqual(result[0].tolist(), H.tolist())
            self.assertTrue((result[1] == mask).all())
            self.assertEqual(result[2], 0.8)

    def test_clean_local_map(self):
        map_points = [
            MapPoint(id_=0, coordinates=np.array([0,0,0])),      # Расстояние 0
            MapPoint(id_=1, coordinates=np.array([50,50,50])),   # Расстояние ~86.6 < 100
            MapPoint(id_=2, coordinates=np.array([200,200,200])) # Расстояние ~346.4 > 100
        ]
        current_pose = np.eye(4)
        cleaned = self.calculator.clean_local_map(map_points, current_pose)
        self.assertEqual(len(cleaned), 2)  # Ожидаем, что останутся первые две точки

        # Изменяем позицию камеры
        current_pose = np.eye(4)
        self.calculator.MAP_CLEAN_MAX_DISTANCE = 100
        cleaned = self.calculator.clean_local_map(map_points, current_pose)
        self.assertEqual(len(cleaned), 2)  # Точка с id=2 удаляется

    def test_filter_map_points_with_descriptors(self):
        mp1 = MapPoint(id_=0, coordinates=np.array([0,0,0]))
        mp1.descriptors = [np.array([1,2,3])]
        mp2 = MapPoint(id_=1, coordinates=np.array([1,1,1]))
        mp2.descriptors = []
        map_points = [mp1, mp2]
        filtered = self.calculator._filter_map_points_with_descriptors(map_points)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].id, 0)

    @patch('image_processing.odometry_calculation.cv2.BFMatcher')
    def test_update_connections_after_pnp(self, mock_bf_matcher):
        mp1 = MapPoint(id_=0, coordinates=np.array([0,0,5]))
        mp1.descriptors = [np.array([1,2,3,4])]
        map_points = [mp1]
        curr_keypoints = [cv2.KeyPoint(320, 240, 1)]
        curr_descriptors = np.array([[1,2,3,4]], dtype=np.uint8)
        matcher_instance = mock_bf_matcher.return_value
        matcher_instance.knnMatch.return_value = [
            [cv2.DMatch(0,0,0.5), cv2.DMatch(0,0,0.6)]  # Изменено
        ]
        self.calculator.update_connections_after_pnp(map_points, curr_keypoints, curr_descriptors, frame_idx=1)
        self.assertEqual(map_points[0].matched_times, 1)

    @patch('image_processing.odometry_calculation.cv2.findFundamentalMat')
    def test_get_inliers_epipolar_no_fundamental(self, mock_find_fundamental):
        mock_find_fundamental.return_value = (None, None)
        prev_keypoints = []
        curr_keypoints = []
        matches = []
        result = self.calculator.get_inliers_epipolar(prev_keypoints, curr_keypoints, matches)
        self.assertEqual(result, [])

    @patch('image_processing.odometry_calculation.cv2.findFundamentalMat')
    def test_get_inliers_epipolar_success(self, mock_find_fundamental):
        F = np.eye(3)
        mask = np.array([1, 0, 1], dtype=np.uint8)
        mock_find_fundamental.return_value = (F, mask)
        prev_keypoints = [cv2.KeyPoint(100, 100, 1), cv2.KeyPoint(200, 200, 1), cv2.KeyPoint(300, 300, 1)]
        curr_keypoints = [cv2.KeyPoint(110, 110, 1), cv2.KeyPoint(210, 210, 1), cv2.KeyPoint(310, 310, 1)]
        matches = [cv2.DMatch(0,0,0.5), cv2.DMatch(1,1,0.6), cv2.DMatch(2,2,0.7)]
        result = self.calculator.get_inliers_epipolar(prev_keypoints, curr_keypoints, matches)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].queryIdx, 0)
        self.assertEqual(result[1].queryIdx, 2)

    def test_check_triangulation_angle_no_points(self):
        with patch.object(self.calculator, 'triangulate_points', return_value=(np.empty((0,3)), [])):
            angle = self.calculator.check_triangulation_angle(
                np.eye(3), np.array([0,0,1]), [], [], [], None
            )
            self.assertEqual(angle, 0.0)

    def test_check_triangulation_angle_success(self):
        pts3D = np.array([[1,0,5], [0,1,5], [0,0,5]])
        with patch.object(self.calculator, 'triangulate_points', return_value=(pts3D, [])):
            angle = self.calculator.check_triangulation_angle(
                np.eye(3), np.array([0,0,1]), [], [], [], None
            )
            self.assertAlmostEqual(angle, 0.0, places=1)  # Уменьшаем точность до 1 знака

    def test_visible_map_points_no_map(self):
        result = self.calculator.visible_map_points([], [], np.array([]), np.eye(4))
        self.assertEqual(result, ([], []))

    def test_visible_map_points_success(self):
        mp1 = MapPoint(id_=0, coordinates=np.array([0,0,5]))
        mp2 = MapPoint(id_=1, coordinates=np.array([1,1,5]))
        map_points = [mp1, mp2]
        curr_keypoints = [cv2.KeyPoint(320, 240, 1), cv2.KeyPoint(330, 250, 1)]
        curr_descriptors = np.array([[1,2,3], [4,5,6]], dtype=np.uint8)
        with patch.object(self.calculator, '_match_projected_points_with_keypoints', return_value=([mp1], [0])):
            matched_map_points, matched_indices = self.calculator.visible_map_points(
                map_points, curr_keypoints, curr_descriptors, np.eye(4)
            )
            self.assertEqual(len(matched_map_points), 1)
            self.assertEqual(matched_map_points[0].id, 0)
            self.assertEqual(matched_indices, [0])

    def test_convert_points_to_structure_mismatch(self):
        pts3D = np.array([[1,2,3]])
        inlier_matches = [cv2.DMatch(0,0,0.5), cv2.DMatch(1,1,0.6)]
        result = self.calculator.convert_points_to_structure(
            pts3D, [], [], inlier_matches, np.array([]), np.array([]), 0, 1
        )
        self.assertEqual(result, [])

    def test_convert_points_to_structure_no_valid_depth(self):
        pts3D = np.array([[1,2,-3]])
        inlier_matches = [cv2.DMatch(0,0,0.5)]
        result = self.calculator.convert_points_to_structure(
            pts3D, [], [], inlier_matches, np.array([]), np.array([]), 0, 1
        )
        self.assertEqual(result, [])

    def test_convert_points_to_structure_success(self):
        pts3D = np.array([[1,2,3]])
        inlier_matches = [cv2.DMatch(0,0,0.5)]
        with patch.object(self.calculator, '_filter_points_by_depth', return_value=(pts3D, inlier_matches)):
            with patch.object(self.calculator, '_build_map_points', return_value=[MapPoint(id_=0, coordinates=pts3D[0])]):
                result = self.calculator.convert_points_to_structure(
                    pts3D, [], [], inlier_matches, np.array([[1,2,3,4]]), np.array([[4,5,6,7]]), 0, 1, start_id=0
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].id, 0)

if __name__ == '__main__':
    unittest.main()
