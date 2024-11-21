import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from image_processing.feature_matching import FeatureMatcher

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MapPoint:
    _id_counter = 0

    def __init__(self, coordinates):
        self.id = MapPoint._id_counter
        MapPoint._id_counter += 1
        self.coordinates = coordinates # 3d coordinates of point
        self.descriptors = [] # list of descriptors of this point
        self.observations = [] # list of observation in frames

class OdometryCalculator:
    def __init__(self, image_width: int, image_height: int, focal_length: Optional[float] = None):
        """
        Инициализация объекта OdometryCalculator.

        Параметры:
        - image_width (int): ширина изображения в пикселях.
        - image_height (int): высота изображения в пикселях.
        - focal_length (float, optional): фокусное расстояние камеры в пикселях.
        """
        self.camera_matrix = self.get_default_camera_matrix(image_width, image_height, focal_length)
        self.dist_coeffs = np.zeros((4, 1))  # Предполагаем отсутствие дисторсии
        logger.info("OdometryCalculator инициализирован с приблизительной матрицей камеры.")

    @staticmethod
    def get_default_camera_matrix(image_width: int, image_height: int, focal_length: Optional[float] = None) -> np.ndarray:
        """
        Создает приблизительную матрицу камеры на основе размеров изображения и фокусного расстояния.

        Параметры:
        - image_width (int): ширина изображения в пикселях.
        - image_height (int): высота изображения в пикселях.
        - focal_length (float, optional): фокусное расстояние в пикселях.

        Возвращает:
        - camera_matrix (numpy.ndarray): матрица внутренней калибровки камеры.
        """
        if focal_length is None:
            focal_length = 0.9 * max(image_width, image_height)  # Коэффициент можно настроить
        cx = image_width / 2
        cy = image_height / 2
        camera_matrix = np.array([[focal_length, 0, cx],
                                  [0, focal_length, cy],
                                  [0, 0, 1]], dtype=np.float64)
        return camera_matrix
    
    def calculate_symmetric_transfer_error(
            self,
            matrix: np.ndarray,
            src_pts: np.ndarray,
            dst_pts: np.ndarray,
            is_homography: bool = False
    ) -> float:
        """
        Вычисляет симметричную ошибку переноса для E или H.

        Параметры:
        - matrix (numpy.ndarray): матрица вращения (3x3).
        - src_pts (np.float32): ключевые точки предыдущего кадра.
        - dst_pts (np.float32): ключевые точки текущего кадра.
        - is_homography (bool): является ли матрица homography

        Возвращает:
        - error (float): симметричная ошибка переноса для матриц Е и Н 
        """
        if is_homography:
            # Преобразуем точки с помощью Homography
            src_pts_h = cv2.convertPointsToHomogeneous(src_pts)[:, 0, :]
            dst_pts_h = cv2.convertPointsToHomogeneous(dst_pts)[:, 0, :]

            src_to_dst = (matrix @ src_pts_h.T).T
            dst_to_src = (np.linalg.inv(matrix) @ dst_pts_h.T).T

            src_to_dst /= src_to_dst[:, 2][:, np.newaxis]
            dst_to_src /= dst_to_src[:, 2][:, np.newaxis]

            error1 = np.linalg.norm(dst_pts - src_to_dst[:, :2], axis=1)
            error2 = np.linalg.norm(src_pts - dst_to_src[:, :2], axis=1)
        else:
            # Используем Essential Matrix
            F = np.linalg.inv(self.camera_matrix).T @ matrix @ np.linalg.inv(self.camera_matrix)
            lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
            lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

            error1 = np.abs(np.sum(src_pts * lines1[:, :2], axis=1) + lines1[:, 2]) / np.linalg.norm(lines1[:, :2], axis=1)
            error2 = np.abs(np.sum(dst_pts * lines2[:, :2], axis=1) + lines2[:, 2]) / np.linalg.norm(lines2[:, :2], axis=1)

        # Возвращаем среднюю симметричную ошибку
        error = np.mean(error1 + error2)
        return error

    def calculate_essential_matrix(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Вычисляет движение между предыдущими и текущими кадрами на основе ключевых точек.

        Параметры:
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - R (numpy.ndarray): матрица вращения (3x3).
        - t (numpy.ndarray): вектор трансляции (3x1).
        - mask (numpy.ndarray): маска с информацией о надежных соответствиях.

        Если движение не может быть вычислено, возвращает None.
        """
        # Определяем соответствующие точки
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Вычисляем матрицу Essential
        E, mask = cv2.findEssentialMat(
            src_pts,
            dst_pts,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            logger.warning("cant calculate Essential.")
            return None
        
        # Вычисляем ошибку переноса
        error = self.calculate_symmetric_transfer_error(E, src_pts, dst_pts)
        return E, mask, error
    
    def calculate_homography_matrix(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        # Получаем соответствующие точки
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Вычисляем матрицу Homography
        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            return None

        # Вычисляем симметричную ошибку переноса
        error = self.calculate_symmetric_transfer_error(H, src_pts, dst_pts, is_homography=True)
        return H, mask, error
    
    def decompose_essential(
        self,
        E: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Получаем соответствующие точки
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Восстанавливаем относительное положение камеры
        _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)
        return R, t, mask_pose

    def decompose_homography(
        self,
        H: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Декомпозируем матрицу Homography
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Получаем возможные решения
        retval, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)
        logger.info(f"Найдено {len(Rs)} возможных решений для Homography.")

        # Выбираем лучшее решение (например, точки перед камерой)
        best_num_inliers = -1
        best_R = None
        best_t = None
        best_mask = None

        for R, t, normal in zip(Rs, ts, normals):
            # Проверяем количество точек перед камерой
            mask = self.check_points_in_front(R, t, src_pts, dst_pts)
            num_inliers = np.sum(mask)
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_R = R
                best_t = t
                best_mask = mask

        return best_R, best_t, best_mask

    def check_points_in_front(self, R, t, src_pts, dst_pts) -> np.ndarray:
        # Триангулируем точки
        proj_matrix1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = self.camera_matrix @ np.hstack((R, t))

        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = pts4D_hom[:3] / pts4D_hom[3]

        # Проверяем, что точки находятся перед камерой
        depths = pts3D[2]
        mask = depths > 0
        return mask

    def check_triangulation_angle(
        self,
        R: np.ndarray,
        t: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> float:
        # Получаем соответствующие точки
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Триангулируем точки
        proj_matrix1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = self.camera_matrix @ np.hstack((R, t))

        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = pts4D_hom[:3] / pts4D_hom[3]

        # Вычисляем базис
        baseline = t.flatten()
        baseline_norm = np.linalg.norm(baseline)

        # Вычисляем углы между лучами и базисом
        angles = []
        for i in range(pts3D.shape[1]):
            point = pts3D[:, i]
            ray = point - np.zeros(3)
            cos_angle = np.dot(ray, baseline) / (np.linalg.norm(ray) * baseline_norm)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)

        median_angle = np.median(angles)
        return median_angle

    def triangulate_points(
        self,
        R: np.ndarray,
        t: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> np.ndarray:
        # Получаем соответствующие точки
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Триангулируем точки
        proj_matrix1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = self.camera_matrix @ np.hstack((R, t))

        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = pts4D_hom[:3] / pts4D_hom[3]
        return pts3D.T

    def convert_points_to_structure(
        self,
        pts3D: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        descriptors: np.ndarray
    ) -> List[MapPoint]:
        """
        Преобразует массив 3D точек в список объектов MapPoint, связывая их с дескрипторами и наблюдениями.
        """
        map_points = []
        for i in range(pts3D.shape[0]):
            mp = MapPoint(pts3D[i])
            # Сохраняем дескриптор, связанный с этой точкой
            descriptor = descriptors[matches[i].trainIdx]
            mp.descriptors.append(descriptor)
            # Сохраняем наблюдение (кадр и ключевую точку)
            mp.observations.append((matches[i].imgIdx, keypoints[matches[i].trainIdx]))
            map_points.append(mp)
        return map_points

    
    def visible_map_points(
            self,
            map_points: List[MapPoint],
            curr_keypoints: List[cv2.KeyPoint],
            curr_descriptors: np.ndarray,
            curr_pose: np.ndarray
    ) -> Tuple[List[MapPoint], List[int]]:
        """
        Находит mappoinst которые находятся в поле зрения текущего кадра и сравниваем их с ключевыми точками
        Параметры:
        - map_points (List[MapPoint]): Список всех map points.
        - curr_keypoints (List[cv2.KeyPoint]): Ключевые точки текущего кадра.
        - curr_pose (np.ndarray): Поза текущей камеры в виде матрицы 3x4 [R|t].

        Возвращает:
        - visible_map_points (List[MapPoint]): Список видимых map points.
        - map_point_indices (List[int]): Индексы ключевых точек, соответствующих видимым map points.
        """
        visible_map_points =[]
        projected_points =[]

        # Разбираем текущую позу камеры
        R_curr = curr_pose[:, :3]
        t_curr = curr_pose[:, 3]

        R_cam = R_curr.T
        t_cam = -R_curr.T @ t_curr

        # Определяем границы изображения
        image_width = self.camera_matrix[0, 2] * 2
        image_height = self.camera_matrix[1, 2] * 2

        for mp in map_points:
            point_world = mp.coordinates.reshape(3, 1)
            point_cam = R_cam @ point_world + t_cam.reshape(3, 1)

            if point_cam[2, 0] <= 0:
                continue

            point_proj = self.camera_matrix @ point_cam
            point_proj /= point_proj[2, 0]

            x, y = point_proj[0, 0], point_proj[1, 0]

            if 0 <= x < image_width and 0 <= y < image_height:
                visible_map_points.append(mp)
                projected_points.append((x, y))

        if len(visible_map_points) == 0:
            self.logger.warning("no visible points")
            return [], []
        
        # Преобразуем ключевые точки текущего кадра в массив координат
        keypoints_coords = np.array([kp.pt for kp in curr_keypoints])

        # Создаем KD-дерево для быстрого поиска ближайших соседей
        from scipy.spatial import cKDTree
        tree = cKDTree(keypoints_coords)

        # Сопоставляем проекции map points с ключевыми точками текущего кадра
        matched_map_points = []
        matched_keypoint_indices = []
        radius = 5  # Порог расстояния для совпадения (в пикселях)


        # Проверяем, что точка в текущем кадре видима
        for idx, (mp, proj_pt) in enumerate((zip(visible_map_points, projected_points))):
            distances, indices = tree.query(proj_pt, k=1, distance_upper_bound=radius)
            if distances != float('inf'):
                keypoint_idx = indices
                
                mp_descriptor = mp.descriptors[0]
                kp_descriptor = curr_descriptors[keypoint_idx]

                distance = cv2.norm(mp_descriptor, kp_descriptor, cv2.NORM_HAMMING)

                if distance < 500:
                    matched_map_points.append(mp)
                    matched_keypoint_indices.append(keypoint_idx)
        
        if len(matched_map_points) == 0:
            logger.warning("no matches")
            return [], []

        return matched_map_points, matched_keypoint_indices
    
    def triangulate_new_map_points(
        self,
        keyframe1: Tuple[int, List[cv2.KeyPoint], np.ndarray, np.ndarray],
        keyframe2: Tuple[int, List[cv2.KeyPoint], np.ndarray, np.ndarray],
        feature_matcher: FeatureMatcher,
        poses: List[np.ndarray]
    ) -> List[MapPoint]:
        """
        Триангулирует новые map points между двумя keyframes.
        Возвращает:
        - new_map_points: Список новых map points.
        """
        idx1, keypoints1, descriptors1, pose1 = keyframe1
        idx2, keypoints2, descriptors2, pose2 = keyframe2

        # Сопоставляем дескрипторы между двумя keyframes
        matches = feature_matcher.match_features(descriptors1, descriptors2)
        if len(matches) < 8:
            logger.warning("Недостаточно совпадений для триангуляции новых map points.")
            return []

        # Получаем соответствующие точки
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])


        R1 = pose1[:, :3]
        t1 = pose1[:, 3]
        R2 = pose2[:, :3]
        t2 = pose2[:, 3]

        # Проекционные матрицы
        proj_matrix1 = self.camera_matrix @ np.hstack((R1, t1.reshape(3, 1)))
        proj_matrix2 = self.camera_matrix @ np.hstack((R2, t2.reshape(3, 1)))

        # Триангулируем точки
        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = (pts4D_hom[:3] / pts4D_hom[3]).T

        # Преобразуем в map points
        new_map_points = []
        for i in range(pts3D.shape[0]):
            mp = MapPoint(pts3D[i])
            # Добавляем дескрипторы и наблюдения
            mp.descriptors.append(descriptors1[matches[i].queryIdx])
            mp.descriptors.append(descriptors2[matches[i].trainIdx])
            mp.observations.append((idx1, keypoints1[matches[i].queryIdx]))
            mp.observations.append((idx2, keypoints2[matches[i].trainIdx]))
            new_map_points.append(mp)

        return new_map_points


