import unittest
from unittest.mock import patch
import itertools
import numpy as np

from image_processing.map_point import MapPoint

class TestMapPoint(unittest.TestCase):
    def setUp(self):
        patcher = patch('src.image_processing.map_point.MapPoint._id_generator', new=itertools.count(0))
        self.addCleanup(patcher.stop)
        self.mock_count = patcher.start()
        
        self.coordinates = np.array([1.0, 2.0, 3.0])
        self.mappoint = MapPoint(coordinates=self.coordinates)
        self.expected_id = 0

    def test_initialization(self):
        self.assertEqual(self.mappoint.id, self.expected_id, f"Expected MapPoint.id to be {self.expected_id}, got {self.mappoint.id}")
        self.assertTrue(np.array_equal(self.mappoint.coordinates, self.coordinates), "MapPoint.coordinates do not match expected values")
        self.assertEqual(len(self.mappoint.descriptors), 0, "MapPoint.descriptors should be empty upon initialization")
        self.assertEqual(len(self.mappoint.observations), 0, "MapPoint.observations should be empty upon initialization")
        self.assertEqual(self.mappoint.matched_times, 0, "MapPoint.matched_times should be 0 upon initialization")

    def test_add_observation(self):
        self.mappoint.add_observation(frame_idx=1, keypoint_idx=2)
        self.assertEqual(len(self.mappoint.observations), 1, "MapPoint.observations should have one entry after add_observation")
        self.assertEqual(self.mappoint.observations[0], (1, 2), "MapPoint.observations content is incorrect after add_observation")

    def test_is_frequently_matched(self):
        self.assertFalse(self.mappoint.is_frequently_matched(), "MapPoint.is_frequently_matched should return False when matched_times < threshold")
        self.mappoint.matched_times = 3
        self.assertTrue(self.mappoint.is_frequently_matched(), "MapPoint.is_frequently_matched should return True when matched_times >= threshold")

    def test_repr(self):
        expected_repr = f"MapPoint(coordinates={self.coordinates}, descriptors=0, observations=0)"
        self.assertEqual(repr(self.mappoint), expected_repr, "MapPoint.__repr__ does not match expected string")

if __name__ == '__main__':
    unittest.main()
