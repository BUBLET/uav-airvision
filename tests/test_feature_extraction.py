import unittest
import numpy as np
from unittest.mock import MagicMock
from src.image_processing.feature_extraction import FeatureExtractor, select_uniform_keypoints_by_grid
import cv2

class TestExtractor:
    def detect(self, image, mask=None):
        return [
            cv2.KeyPoint(x=10, y=10, size=1),
            cv2.KeyPoint(x=20, y=20, size=1)
        ]
    
    def compute(self, image, keypoints):
        return keypoints, np.array([[1, 2, 3], [4, 5, 6]])

class TestFeatureExtractor(unittest.TestCase):
    def test_extract_features(self):
        # Заготовка тестового изображения (черное изображение)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        fake_extractor = TestExtractor()

        feature_extractor = FeatureExtractor(extractor=fake_extractor, grid_size=10, max_pts_per_cell=2)

        keypoints, descriptors = feature_extractor.extract_features(test_image)

        self.assertEqual(len(keypoints), 2)

        self.assertAlmostEqual(keypoints[0].pt[0], 10)
        self.assertAlmostEqual(keypoints[0].pt[1], 10)
        self.assertAlmostEqual(keypoints[1].pt[0], 20)
        self.assertAlmostEqual(keypoints[1].pt[1], 20)

        expected_descriptors = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(descriptors, expected_descriptors)

    def test_extract_features_empty_image(self):
        feature_extractor = FeatureExtractor(extractor=TestExtractor())

        with self.assertRaises(ValueError):
            feature_extractor.extract_features(None)

        with self.assertRaises(ValueError):
            feature_extractor.extract_features(np.array([]))

    def test_select_uniform_keypoints_by_grid(self):
        keypoints = [
            cv2.KeyPoint(x=5, y=5, size=1),
            cv2.KeyPoint(x=15, y=15, size=1),
            cv2.KeyPoint(x=25, y=25, size=1),
            cv2.KeyPoint(x=35, y=35, size=1)
        ]
        image_rows = 40
        image_cols = 40
        grid_size = 10
        max_pts_per_cell = 1

        selected_keypoints = select_uniform_keypoints_by_grid(
            keypoints, image_rows, image_cols, grid_size, max_pts_per_cell
        )

        self.assertEqual(len(selected_keypoints), 4)

        self.assertEqual(selected_keypoints[0].pt, (5, 5))
        self.assertEqual(selected_keypoints[1].pt, (15, 15))
        self.assertEqual(selected_keypoints[2].pt, (25, 25))
        self.assertEqual(selected_keypoints[3].pt, (35, 35))
    
    def test_select_uniform_keypoints_by_grid_max_points(self):
        keypoints = [
            cv2.KeyPoint(x=5, y=5, size=1),
            cv2.KeyPoint(x=6, y=5, size=1),
            cv2.KeyPoint(x=7, y=5, size=1),
            cv2.KeyPoint(x=15, y=15, size=1)
        ]
        image_rows = 20
        image_cols = 20
        grid_size = 10
        max_pts_per_cell = 2

        selected_keypoints = select_uniform_keypoints_by_grid(
            keypoints, image_rows, image_cols, grid_size, max_pts_per_cell
        )


        self.assertEqual(len(selected_keypoints), 3)
        self.assertIn(selected_keypoints[0].pt, [(5, 5), (6, 5), (7, 5)])
        self.assertIn(selected_keypoints[1].pt, [(5, 5), (6, 5), (7, 5)])
        self.assertEqual(selected_keypoints[2].pt, (15, 15))


if __name__ == '__main__':
    unittest.main()
