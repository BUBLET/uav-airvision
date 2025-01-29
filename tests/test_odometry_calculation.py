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
        # Сбрасываем генератор идентификаторов перед каждым тестом
        MapPoint._id_generator = itertools.count(0)
        
        # Инициализируем OdometryCalculator с необходимыми параметрами
        self.camera_matrix = np.array([[800, 0, 320],
                                       [0, 800, 240],
                                       [0, 0, 1]], dtype=np.float64)
        self.calculator = OdometryCalculator(
            image_width=640,
            image_height=480,
            camera_matrix=self.camera_matrix,
            logger=MagicMock()
        )
        
        # Создаем фиктивные ключевые точки с правильным использованием cv2.KeyPoint
        self.prev_keypoints = [cv2.KeyPoint(i*10, i*10, 1) for i in range(5)]
        self.curr_keypoints = [cv2.KeyPoint(i*10+5, i*10+5, 1) for i in range(5)]
        
        # Создаем фиктивные дескрипторы
        self.prev_descriptors = np.random.randint(0, 256, (5, 32), dtype=np.uint8)
        self.curr_descriptors = np.random.randint(0, 256, (5, 32), dtype=np.uint8)
        
        # Создаем фиктивные матчи с достаточным количеством совпадений для гомографии (не менее 4)
        self.homography_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.5) for i in range(5)]  # 5 matches

        # Создаем фиктивные матчи для триангуляции (может быть менее 8, так как тестируем конкретные случаи)
        self.triangulation_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.5) for i in range(3)]  # 3 matches

    def test_calculate_homography_matrix_success(self):
        # Используем 4 matches для гомографии
        matches = self.homography_matches[:4]
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

    def test_visible_map_points_success(self):
        # Создаем фиктивные MapPoints, проецируемые внутри изображения
        mp1 = MapPoint(coordinates=np.array([0, 0, 5]))    # Проецируется в (320, 240)
        mp2 = MapPoint(coordinates=np.array([100, 100, 5]))  # Проецируется в (420, 340)
        map_points = [mp1, mp2]

        # Создаем матрицу позы с размерностью (3,4)
        curr_pose = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]], dtype=np.float64)
        
        matched_map_points, matched_indices = self.calculator.visible_map_points(
            map_points,
            self.curr_keypoints,
            self.curr_descriptors,
            curr_pose
        )
        self.assertEqual(len(matched_map_points), 2, "Number of matched map points is incorrect")
        self.assertEqual(len(matched_indices), 2, "Number of matched indices is incorrect")

    def test_convert_points_to_structure_success(self):
        # Создаем триангулированные 3D-точки
        pts3D = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        inlier_matches = [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.5)]
       
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
        self.assertEqual(len(result), 1, "Number of MapPoints created is incorrect")
        self.assertEqual(result[0].id, 0, "MapPoint.id is incorrect")
        self.assertEqual(result[0].matched_times, 0, "MapPoint.matched_times is incorrect")
        self.assertEqual(len(result[0].descriptors), 2, "Number of descriptors in MapPoint is incorrect")
        self.assertEqual(len(result[0].observations), 2, "Number of observations in MapPoint is incorrect")

    def test_filter_map_points_with_descriptors(self):
        # Создаем MapPoints с и без дескрипторов
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

    def test_get_inliers_epipolar_success(self):
        # Патчим cv2.findFundamentalMat, чтобы всегда возвращать все inliers
        with patch('cv2.findFundamentalMat', return_value=(np.eye(3), np.ones((len(self.homography_matches[:3]), 1), dtype=np.uint8))):
            result = self.calculator.get_inliers_epipolar(
                self.prev_keypoints,
                self.curr_keypoints,
                self.homography_matches[:3]  # Используем первые 3 матча
            )
            self.assertEqual(len(result), 3, "Number of inliers is incorrect")

    def test_triangulate_new_map_points_success(self):
        # В OdometryCalculator, метод triangulate_new_map_points не использует '_calculate_symmetric_transfer_error'
        # Поэтому патчить этот метод не требуется.

        keyframe1 = (0, self.prev_keypoints, self.prev_descriptors, np.hstack((np.eye(3), np.zeros((3,1)))))
        keyframe2 = (1, self.curr_keypoints, self.curr_descriptors, np.hstack((np.eye(3), np.array([[0], [0], [1]]))))
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.5) for i in range(3)]

        # Не патчим ничего, так как triangulate_new_map_points не вызывает '_calculate_symmetric_transfer_error'
        new_map_points = self.calculator.triangulate_new_map_points(
            keyframe1,
            keyframe2,
            matches
        )
        
        self.assertEqual(len(new_map_points), 3, "Number of triangulated MapPoints is incorrect")
        self.assertEqual(new_map_points[0].id, 0, "First triangulated MapPoint.id is incorrect")
        self.assertEqual(new_map_points[1].id, 1, "Second triangulated MapPoint.id is incorrect")
        self.assertEqual(new_map_points[2].id, 2, "Third triangulated MapPoint.id is incorrect")
        self.assertEqual(new_map_points[0].matched_times, 0, "MapPoint.matched_times is incorrect")
        self.assertEqual(len(new_map_points[0].descriptors), 2, "MapPoint.descriptors count is incorrect")
        self.assertEqual(len(new_map_points[0].observations), 2, "MapPoint.observations count is incorrect")

    def test_update_connections_after_pnp(self):
        # Создаем MapPoints с дескрипторами
        mp1 = MapPoint(coordinates=np.array([0.5, 0.5, 1.0]))
        mp1.descriptors = [self.prev_descriptors[0], self.curr_descriptors[0]]
        
        map_points = [mp1]
        
        # Создаем текущие ключевые точки и дескрипторы
        curr_keypoints = [cv2.KeyPoint(5.0, 5.0, 1)]
        curr_descriptors = np.array([self.curr_descriptors[0]], dtype=np.uint8)
        
        # Патчим методы сопоставления и фильтрации
        with patch.object(self.calculator, '_knn_match', return_value=[[cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.5)]]):
            with patch.object(self.calculator, '_lowe_ratio_test', return_value=[cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.5)]):
                self.calculator.update_connections_after_pnp(
                    map_points,
                    curr_keypoints,
                    curr_descriptors,
                    frame_idx=1
                )
        
        self.assertEqual(map_points[0].matched_times, 1, "MapPoint.matched_times was not updated correctly")

if __name__ == '__main__':
    unittest.main()
