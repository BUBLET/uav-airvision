import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, Mock

from src.image_processing.feature_matching import FeatureMatcher

class MockMatcher:
    def knnMatch(self, queryDescriptors, trainDescriptors, k):
        if len(queryDescriptors) == 0 or len(trainDescriptors) == 0:
            return []
        matches = []
        for i in range(len(queryDescriptors)):
            inner_matches = []
            for j in range(min(k, len(trainDescriptors))):
                dmatch = cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0.1 * (j + 1))
                inner_matches.append(dmatch)
            matches.append(inner_matches)
        return matches

class TestFeatureMatcher(unittest.TestCase):
    def setUp(self):
        self.mock_matcher = MagicMock()
        self.feature_matcher = FeatureMatcher(
            matcher=self.mock_matcher,
            knn_k=2,
            lowe_ratio=0.75
        )
    
    def test_match_feature_success(self):

        descriptors1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        descriptors2 = np.array([[1, 2, 3], [7, 8, 9]], dtype=np.uint8)

        # Настраиваем мок-объект matcher.knnMatch
        mock_matches = [
            [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.1),
             cv2.DMatch(_queryIdx=0, _trainIdx=1, _distance=0.2)],
            [cv2.DMatch(_queryIdx=1, _trainIdx=0, _distance=0.1),
             cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=0.2)]
        ]
        self.mock_matcher.knnMatch.return_value = mock_matches

        # Вызываем метод
        good_matches = self.feature_matcher.match_features(descriptors1, descriptors2)

        # Ожидаемые результаты: оба m.distance < lowe_ratio * n.distance
        expected_matches = [mock_matches[0][0], mock_matches[1][0]]

        # Проверяем результаты
        self.assertEqual(len(good_matches), 2)
        self.assertEqual(good_matches, expected_matches)
        self.mock_matcher.knnMatch.assert_called_once_with(descriptors1, descriptors2, k=2)
    
    def test_match_features_no_good_matches(self):
       
        descriptors1 = np.array([[1, 2, 3]], dtype=np.uint8)
        descriptors2 = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        
        self.mock_matcher.knnMatch = lambda q, t, k: [
            [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.8),
             cv2.DMatch(_queryIdx=0, _trainIdx=1, _distance=1.0)]
        ]
        
        good_matches = self.feature_matcher.match_features(descriptors1, descriptors2)
        
        # Ожидаем, что good_matches будет пустым
        self.assertEqual(len(good_matches), 0)
    
    def test_match_features_empty_descriptors(self):

        descriptors1 = np.array([], dtype=np.uint8).reshape(0, 32)
        descriptors2 = np.array([], dtype=np.uint8).reshape(0, 32)
        
        
        with self.assertLogs('src.image_processing.feature_matching', level='WARNING') as log:
            good_matches = self.feature_matcher.match_features(descriptors1, descriptors2)
            self.assertEqual(len(good_matches), 0)
            # Проверяем, что в логах есть предупреждение
            self.assertIn('WARNING:src.image_processing.feature_matching:descriptors are empty', log.output)
    
    def test_match_features_none_descriptors(self):

        with self.assertRaises(ValueError):
            self.feature_matcher.match_features(None, None)
        
        with self.assertRaises(ValueError):
            self.feature_matcher.match_features(np.array([]), None)
        
        with self.assertRaises(ValueError):
            self.feature_matcher.match_features(None, np.array([]))
    
    def test_match_features_no_matches_found(self):

        descriptors1 = np.array([[1, 2, 3]], dtype=np.uint8)
        descriptors2 = np.array([[4, 5, 6]], dtype=np.uint8)
        
        # Настраиваем мок, чтобы он возвращал пустой список
        self.mock_matcher.knnMatch = MagicMock(return_value=[])
        
        with self.assertLogs('src.image_processing.feature_matching', level='WARNING') as log:
            good_matches = self.feature_matcher.match_features(descriptors1, descriptors2)
            self.assertEqual(len(good_matches), 0)
            self.assertIn('WARNING:src.image_processing.feature_matching:no matches', log.output)
    
if __name__ == '__main__':
    unittest.main()