import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from src.image_processing.frame_processor import FrameProcessor

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.feature_extractor_mock = MagicMock()
        self.feature_matcher_mock = MagicMock()
        self.odometry_calculator_mock = MagicMock()
        self.frame_processor = FrameProcessor(
            feature_extractor=self.feature_extractor_mock,
            feature_matcher=self.feature_matcher_mock,
            odometry_calculator=self.odometry_calculator_mock,
            translation_threshold=0.1,
            rotation_threshold=0.1,
            triangulation_threshold=0.01,
            bundle_adjustment_frames=2,
            force_keyframe_interval=1,
            homography_inlier_ratio=0.6
        )

    def test_initialize_map_with_few_matches(self):
        self.feature_matcher_mock.match_features.return_value = []
        result = self.frame_processor._initialize_map([], np.array([]), [], np.array([]))
        self.assertIsNone(result)

    def test_initialize_map_no_E_or_H(self):
        self.feature_matcher_mock.match_features.return_value = [MagicMock()] * 10
        self.odometry_calculator_mock.calculate_essential_matrix.return_value = None
        self.odometry_calculator_mock.calculate_homography_matrix.return_value = None
        result = self.frame_processor._initialize_map([], np.array([]), [], np.array([]))
        self.assertIsNone(result)

    def test_initialize_map_with_valid_data_and_low_median_angle(self):
        self.feature_matcher_mock.match_features.return_value = [MagicMock()] * 10
        self.odometry_calculator_mock.calculate_essential_matrix.return_value = (np.eye(3), np.ones((10,1)), 0.1)
        self.odometry_calculator_mock.calculate_homography_matrix.return_value = (np.eye(3), np.ones((10,1)), 0.2)
        self.odometry_calculator_mock.decompose_essential.return_value = (np.eye(3), np.array([0,0,1]), np.ones((10,1)))
        self.odometry_calculator_mock.check_triangulation_angle.return_value = 0.0001
        result = self.frame_processor._initialize_map([], np.array([]), [], np.array([]))
        self.assertIsNone(result)

    def test_initialize_map_success(self):
        self.feature_matcher_mock.match_features.return_value = [MagicMock()] * 10
        self.odometry_calculator_mock.calculate_essential_matrix.return_value = (np.eye(3), np.ones((10,1)), 0.1)
        self.odometry_calculator_mock.calculate_homography_matrix.return_value = (np.eye(3), np.ones((10,1)), 0.2)
        self.odometry_calculator_mock.decompose_essential.return_value = (np.eye(3), np.array([0,0,1]), np.ones((10,1)))
        self.odometry_calculator_mock.check_triangulation_angle.return_value = 0.5
        result = self.frame_processor._initialize_map([], np.array([]), [], np.array([]))
        self.assertIsNotNone(result)

    def test_triangulate_initial_points(self):
        self.odometry_calculator_mock.triangulate_points.return_value = (
            np.random.rand(10,3), 
            [MagicMock()] * 10
        )
        self.odometry_calculator_mock.convert_points_to_structure.return_value = ["mp1", "mp2"]
        map_points = []
        result = self.frame_processor._triangulate_initial_points(
            np.eye(3), 
            np.array([0,0,1]), 
            [], 
            [], 
            np.array([]), 
            np.array([]), 
            [], 
            np.ones((10,1)), 
            1, 
            map_points
        )
        self.assertEqual(result, ["mp1", "mp2"])

    def test_estimate_pose_not_enough_visible_points(self):
        self.odometry_calculator_mock.visible_map_points.return_value = ([], [])
        result = self.frame_processor._estimate_pose([], [], np.array([]), np.eye(3,4))
        self.assertIsNone(result)

    def test_estimate_pose_success(self):
        mock_map_points = [MagicMock(coordinates=np.array([1.0, 2.0, 3.0])) for _ in range(8)]
        self.odometry_calculator_mock.visible_map_points.return_value = (
            mock_map_points, list(range(8))
        )
        
        # Создаём 8 ключевых точек с атрибутом .pt
        mock_keypoints = [MagicMock(pt=(i, i)) for i in range(8)]
        
        with patch('cv2.solvePnPRansac', return_value=(
            True, 
            np.array([0.1, 0.2, 0.3]), 
            np.array([[1.0], [2.0], [3.0]]),  # Исправлено: tvec как 2D массив
            np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
        )):
            result = self.frame_processor._estimate_pose(
                map_points=mock_map_points,
                curr_keypoints=mock_keypoints,
                curr_descriptors=np.array([]),
                last_pose=np.eye(3, 4)
            )
            self.assertIsNotNone(result)

    def test_should_insert_keyframe_forced(self):
        current_pose = np.eye(3,4)
        last_keyframe_pose = np.eye(3,4)
        result = self.frame_processor._should_insert_keyframe(last_keyframe_pose, current_pose, 2)
        self.assertTrue(result)

    def test_should_insert_keyframe_by_threshold(self):
        current_pose = np.eye(4)
        last_keyframe_pose = np.eye(4)
        current_pose[:3,3] = np.array([0.2,0,0])
        current_pose = current_pose[:3, :4]
        result = self.frame_processor._should_insert_keyframe(last_keyframe_pose, current_pose, 3)
        self.assertTrue(result)

    def test_insert_keyframe_insufficient_inliers(self):
        keyframes = [(0, [], np.array([]), np.eye(3,4))]
        self.feature_matcher_mock.match_features.return_value = [MagicMock()] * 5
        self.odometry_calculator_mock.get_inliers_epipolar.return_value = []
        map_points = []
        self.frame_processor._insert_keyframe(1, [], np.array([]), np.eye(3,4), keyframes, map_points)
        self.assertEqual(len(keyframes), 2)

    def test_insert_keyframe_triangulation(self):
        keyframes = [(0, [], np.array([]), np.eye(3,4))]
        self.feature_matcher_mock.match_features.return_value = [MagicMock()] * 20
        self.odometry_calculator_mock.get_inliers_epipolar.return_value = [MagicMock()] * 10
        self.odometry_calculator_mock.triangulate_new_map_points.return_value = ["p1", "p2"]
        map_points = []
        self.frame_processor._insert_keyframe(1, [], np.array([]), np.eye(3,4), keyframes, map_points)
        self.assertEqual(len(map_points), 2)

    def test_run_local_ba_no_data(self):
        self.frame_processor._collect_ba_data = MagicMock(return_value=None)
        self.frame_processor._run_local_ba([], [])
        self.assertTrue(True)

    def test_run_local_ba_success(self):
        self.frame_processor._collect_ba_data = MagicMock(
            return_value=(
                np.zeros((2,6)), 
                np.zeros((5,3)), 
                np.array([0,1,1,0,0]), 
                np.array([0,1,2,3,4]), 
                np.random.rand(5,2)
            )
        )
        mock_ba = MagicMock()
        with patch('src.image_processing.frame_processor.BundleAdjustment', return_value=mock_ba):
            mock_ba.run_bundle_adjustment.return_value = (
                np.ones((2,6)), 
                np.ones((5,3))
            )
            keyframes = [
                (0, [], np.array([]), np.eye(3,4)), 
                (1, [], np.array([]), np.eye(3,4))
            ]
            map_points = [MagicMock(id=i, coordinates=np.zeros(3)) for i in range(5)]
            self.frame_processor._update_optimized_values = MagicMock()
            self.frame_processor._run_local_ba(keyframes, map_points)
            self.assertTrue(self.frame_processor._update_optimized_values.called)

    def test_clean_local_map(self):
        self.odometry_calculator_mock.clean_local_map.return_value = ["new_map"]
        result = self.frame_processor._clean_local_map(["old_map"], np.eye(3,4))
        self.assertEqual(result, ["new_map"])

    def test_process_frame_initialization_first_call(self):
        self.feature_extractor_mock.extract_features.return_value = (["kp1"], np.array([1]))
        result = self.frame_processor.process_frame(
            frame_idx=0, 
            current_frame=MagicMock(), 
            ref_keypoints=None, 
            ref_descriptors=None, 
            last_pose=np.eye(3,4), 
            map_points=[], 
            initialization_completed=False, 
            poses=[], 
            keyframes=[]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], ["kp1"])

    def test_process_frame_pose_estimation_failed(self):
        self.feature_extractor_mock.extract_features.return_value = (["kp1"], np.array([1]))
        self.odometry_calculator_mock.visible_map_points.return_value = ([], [])
        result = self.frame_processor.process_frame(
            frame_idx=2, 
            current_frame=MagicMock(), 
            ref_keypoints=["kp0"], 
            ref_descriptors=np.array([1]), 
            last_pose=np.eye(3,4), 
            map_points=[], 
            initialization_completed=True, 
            poses=[], 
            keyframes=[(0, ["kp0"], np.array([1]), np.eye(3,4))]
        )
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
