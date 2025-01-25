import numpy as np
import unittest
from src.image_processing import MapPoint

class TestMapPoint(unittest.TestCase):
    
    def setUp(self):
        self.id = 1
        self.coordinates = np.array([1.0, 2.0, 3.0])
        self.mappoint = MapPoint(self.id, self.coordinates)
    
    def test_initialization(self):
        self.assertEqual(self.mappoint.id, self.id)
        np.testing.assert_array_equal(self.mappoint.coordinates, self.coordinates)
        self.assertEqual(self.mappoint.descriptors, [])
        self.assertEqual(self.mappoint.observations, [])
        self.assertEqual(self.mappoint.matched_times, 0)
    
    def test_add_observation(self):
        self.mappoint.add_observation(frame_idx=1, keypoint_idx=101)
        self.mappoint.add_observation(frame_idx=2, keypoint_idx=202)
        self.assertEqual(len(self.mappoint.observations), 2)
        self.assertIn((1, 101), self.mappoint.observations)
        self.assertIn((2, 202), self.mappoint.observations)
    
    def test_is_frequently_matched(self):
        # Проверяем метод is_frequently_matched с порогом по умолчанию
        self.assertFalse(self.mappoint.is_frequently_matched())
        
        # Вручную устанавливаем matched_times для тестирования
        self.mappoint.matched_times = 2
        self.assertFalse(self.mappoint.is_frequently_matched())
        
        self.mappoint.matched_times = 3
        self.assertTrue(self.mappoint.is_frequently_matched())
        
        # Проверяем с другим порогом
        self.assertTrue(self.mappoint.is_frequently_matched(threshold=2))
        self.assertFalse(self.mappoint.is_frequently_matched(threshold=4))
    
    def test_repr(self):
        self.mappoint.add_observation(1, 101)
        self.mappoint.add_observation(2, 202)
        repr_str = repr(self.mappoint)
        expected_str = f"MapPoint(coordinates={self.coordinates}, descriptors=0, observations=2)"
        self.assertEqual(repr_str, expected_str)
    
    def test_descriptors(self):
        descriptor1 = np.array([0.1, 0.2, 0.3])
        descriptor2 = np.array([0.4, 0.5, 0.6])
        self.mappoint.descriptors.append(descriptor1)
        self.mappoint.descriptors.append(descriptor2)
        self.assertEqual(len(self.mappoint.descriptors), 2)
        np.testing.assert_array_equal(self.mappoint.descriptors[0], descriptor1)
        np.testing.assert_array_equal(self.mappoint.descriptors[1], descriptor2)


if __name__ == '__main__':
    unittest.main()