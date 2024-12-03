import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MapPoint:
    _id_counter = 0

    def __init__(self, coordinates):
        self.id = MapPoint._id_counter
        MapPoint._id_counter += 1
        self.coordinates = coordinates  # 3D координаты точки
        self.descriptors = []  # Список дескрипторов этой точки
        self.observations = []  # Список наблюдений в кадрах
        self.matched_times = 0

    def add_observation(self, frame_idx, keypoint_idx):
        self.observations.append((frame_idx, keypoint_idx))

    def is_frequently_matched(self, threshold=3):
        return self.matched_times >= threshold

    def __repr__(self):
        return f"MapPoint(coordinates={self.coordinates}, descriptors={len(self.descriptors)}, observations={len(self.observations)})"

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
        self.logger = logging.getLogger(__name__)
        self.logger.info("OdometryCalculator инициализирован с приблизительной матрицей камеры.")

    @staticmethod
    def get_default_camera_matrix(image_width: int, image_height: int, focal_length: Optional[float] = None) -> np.ndarray:
        """
        Создает матрицу камеры на основе известных параметров для набора данных 'matlab'.

        Параметры:
        - image_width (int): ширина изображения в пикселях.
        - image_height (int): высота изображения в пикселях.
        - focal_length (float, optional): фокусное расстояние в миллиметрах (не используется, но оставлено для совместимости).

        Возвращает:
        - camera_matrix (numpy.ndarray): матрица внутренней калибровки камеры.
        """
        # Параметры матрицы камеры для набора данных 'matlab'
        fx = 615.0  # Фокусное расстояние по оси x в пикселях
        fy = 615.0  # Фокусное расстояние по оси y в пикселях
        cx = 320.0  # Главная точка по оси x (центр изображения)
        cy = 240.0  # Главная точка по оси y (центр изображения)

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
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
        - matrix (numpy.ndarray): матрица вращения или Homography (3x3).
        - src_pts (np.ndarray): ключевые точки предыдущего кадра.
        - dst_pts (np.ndarray): ключевые точки текущего кадра.
        - is_homography (bool): является ли матрица Homography.

        Возвращает:
        - error (float): симметричная ошибка переноса.
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
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Вычисляет матрицу Essential между предыдущими и текущими кадрами на основе ключевых точек.

        Параметры:
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - E (numpy.ndarray): матрица Essential (3x3).
        - mask (numpy.ndarray): маска с информацией о надежных соответствиях.
        - error (float): симметричная ошибка переноса.

        Если матрицу Essential не удалось вычислить, возвращает None.
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
            self.logger.warning("Не удалось вычислить матрицу Essential.")
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
        """
        Вычисляет матрицу Homography между предыдущими и текущими кадрами на основе ключевых точек.

        Параметры:
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - H (numpy.ndarray): матрица Homography (3x3).
        - mask (numpy.ndarray): маска с информацией о надежных соответствиях.
        - error (float): симметричная ошибка переноса.

        Если матрицу Homography не удалось вычислить, возвращает None.
        """
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
        """
        Декомпозирует матрицу Essential и восстанавливает относительное движение между двумя наборами точек.

        Параметры:
        - E (numpy.ndarray): матрица Essential.
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - R (numpy.ndarray): матрица вращения (3x3).
        - t (numpy.ndarray): вектор трансляции (3x1).
        - mask_pose (numpy.ndarray): маска инлайеров, используемых для восстановления позы.
        """
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
        """
        Декомпозирует матрицу Homography и выбирает лучшее решение на основе числа точек перед камерой.

        Параметры:
        - H (numpy.ndarray): матрица Homography.
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - best_R (numpy.ndarray): лучшая матрица вращения.
        - best_t (numpy.ndarray): лучший вектор трансляции.
        - best_mask (numpy.ndarray): маска инлайеров для лучшего решения.
        """
        # Декомпозируем матрицу Homography
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Получаем возможные решения
        retval, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)
        self.logger.info(f"Найдено {len(Rs)} возможных решений для Homography.")

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
        """
        Проверяет, находятся ли триангулированные точки перед камерой.

        Параметры:
        - R (numpy.ndarray): матрица вращения.
        - t (numpy.ndarray): вектор трансляции.
        - src_pts (np.ndarray): точки из предыдущего кадра.
        - dst_pts (np.ndarray): точки из текущего кадра.

        Возвращает:
        - mask (np.ndarray): маска точек, находящихся перед камерой.
        """
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
        """
        Вычисляет медианный угол триангуляции между лучами и базисом камеры.

        Параметры:
        - R (numpy.ndarray): матрица вращения.
        - t (numpy.ndarray): вектор трансляции.
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.

        Возвращает:
        - median_angle (float): медианный угол триангуляции в радианах.
        """
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
            ray = point  # ray = point - np.zeros(3)
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
        matches: List[cv2.DMatch],
        mask_pose: np.ndarray
    ) -> Tuple[np.ndarray, List[cv2.DMatch]]:
        """
        Триангулирует 3D точки из соответствующих 2D точек двух кадров.

        Параметры:
        - R (numpy.ndarray): матрица вращения.
        - t (numpy.ndarray): вектор трансляции.
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - matches (list of cv2.DMatch): список сопоставленных точек.
        - mask_pose (np.ndarray): маска инлайеров, используемых для восстановления позы.

        Возвращает:
        - pts3D (np.ndarray): массив триангулированных 3D точек.
        - inlier_matches (list of cv2.DMatch): список инлайерных сопоставлений.
        """
        # Используем только инлайеры
        inlier_matches = [matches[i] for i in range(len(matches)) if mask_pose[i]]
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in inlier_matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in inlier_matches])

        # Проекционные матрицы
        proj_matrix1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = self.camera_matrix @ np.hstack((R, t))

        # Триангулируем точки
        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = (pts4D_hom[:3] / pts4D_hom[3]).T
        return pts3D, inlier_matches

    def convert_points_to_structure(
        self,
        pts3D: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch],
        prev_descriptors: np.ndarray,
        curr_descriptors: np.ndarray,
        prev_frame_idx: int,
        curr_frame_idx: int
    ) -> List[MapPoint]:
        """
        Преобразует триангулированные 3D точки в объекты MapPoint.

        Параметры:
        - pts3D (np.ndarray): массив триангулированных 3D точек.
        - prev_keypoints (list of cv2.KeyPoint): ключевые точки предыдущего кадра.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - inlier_matches (list of cv2.DMatch): список инлайерных сопоставлений.
        - prev_descriptors (np.ndarray): дескрипторы предыдущего кадра.
        - curr_descriptors (np.ndarray): дескрипторы текущего кадра.
        - prev_frame_idx (int): индекс предыдущего кадра.
        - curr_frame_idx (int): индекс текущего кадра.

        Возвращает:
        - map_points (list of MapPoint): список созданных точек карты.
        """
        map_points = []
        assert len(pts3D) == len(inlier_matches), "Количество 3D-точек не совпадает с количеством соответствий"
        for i in range(len(pts3D)):
            if pts3D[i][2] <= 0:
                self.logger.warning(f"Точка {i} имеет отрицательную глубину: {pts3D[i]}")
                continue  # Пропустить эту точку
            mp = MapPoint(pts3D[i])
            # Сохраняем дескрипторы из предыдущего и текущего кадров
            descriptor_prev = prev_descriptors[inlier_matches[i].queryIdx]
            descriptor_curr = curr_descriptors[inlier_matches[i].trainIdx]
            mp.descriptors.extend([descriptor_prev, descriptor_curr])
            # Сохраняем наблюдения
            mp.observations.extend([
                (prev_frame_idx, inlier_matches[i].queryIdx),
                (curr_frame_idx, inlier_matches[i].trainIdx)
            ])
            map_points.append(mp)
            self.logger.debug(f"Добавлена новая точка карты {i}: {mp.coordinates}")
        self.logger.info(f"Всего добавлено {len(map_points)} новых точек карты.")
        return map_points

    def visible_map_points(
        self,
        map_points: List[MapPoint],
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray,
        curr_pose: np.ndarray
    ) -> Tuple[List[MapPoint], List[int]]:
        """
        Определяет видимые точки карты в текущем кадре и сопоставляет их с ключевыми точками.

        Параметры:
        - map_points (list of MapPoint): список точек карты.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - curr_descriptors (np.ndarray): дескрипторы текущего кадра.
        - curr_pose (np.ndarray): текущая поза камеры.

        Возвращает:
        - matched_map_points (list of MapPoint): сопоставленные точки карты.
        - matched_keypoint_indices (list of int): индексы ключевых точек, соответствующих точкам карты.
        """
        visible_map_points = []
        projected_points = []

        # Разбираем текущую позу камеры
        R_curr = curr_pose[:, :3]
        t_curr = curr_pose[:, 3].reshape(3, 1)

        # Определяем границы изображения
        image_width = self.camera_matrix[0, 2] * 2
        image_height = self.camera_matrix[1, 2] * 2

        self.logger.info(f"Total map points: {len(map_points)}")

        for mp in map_points:
            point_world = mp.coordinates.reshape(3, 1)
            # Преобразуем точку из мировой системы координат в систему координат камеры
            point_cam = R_curr @ point_world + t_curr

            if point_cam[2, 0] <= 0:
                continue

            point_proj = self.camera_matrix @ point_cam
            point_proj /= point_proj[2, 0]

            x, y = point_proj[0, 0], point_proj[1, 0]

            if 0 <= x < image_width and 0 <= y < image_height:
                visible_map_points.append(mp)
                projected_points.append((x, y))

        self.logger.info(f"Number of projected map points: {len(projected_points)}")

        if len(visible_map_points) == 0:
            self.logger.warning("Нет видимых точек")
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

        for idx, (mp, proj_pt) in enumerate(zip(visible_map_points, projected_points)):
            distances, indices = tree.query(proj_pt, k=1, distance_upper_bound=radius)
            if distances != float('inf'):
                keypoint_idx = indices

                mp_descriptor = mp.descriptors[-1]  # Используем последний дескриптор
                kp_descriptor = curr_descriptors[keypoint_idx]

                distance = cv2.norm(mp_descriptor, kp_descriptor, cv2.NORM_HAMMING)

                if distance < 100:
                    matched_map_points.append(mp)
                    matched_keypoint_indices.append(keypoint_idx)

        if len(matched_map_points) == 0:
            self.logger.warning("Нет совпадений")
            return [], []

        self.logger.info(f"Number of matched map points: {len(matched_map_points)}")
        return matched_map_points, matched_keypoint_indices

    def triangulate_new_map_points(
        self,
        keyframe1: Tuple[int, List[cv2.KeyPoint], np.ndarray, np.ndarray],
        keyframe2: Tuple[int, List[cv2.KeyPoint], np.ndarray, np.ndarray],
        inlier_matches: List[cv2.DMatch]
    ) -> List[MapPoint]:
        """
        Триангулирует новые точки карты между двумя кейфреймами.

        Параметры:
        - keyframe1 (tuple): кортеж (индекс кадра, ключевые точки, дескрипторы, поза).
        - keyframe2 (tuple): кортеж (индекс кадра, ключевые точки, дескрипторы, поза).
        - inlier_matches (list of cv2.DMatch): список инлайерных сопоставлений.

        Возвращает:
        - new_map_points (list of MapPoint): список новых точек карты.
        """
        idx1, keypoints1, descriptors1, pose1 = keyframe1
        idx2, keypoints2, descriptors2, pose2 = keyframe2

        # Проверяем количество совпадений
        if len(inlier_matches) < 8:
            self.logger.warning("Недостаточно совпадений для триангуляции новых map points.")
            return []

        # Получаем соответствующие точки
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in inlier_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in inlier_matches])

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
            if pts3D[i][2] <= 0:
                self.logger.warning(f"Точка {i} имеет отрицательную глубину: {pts3D[i]}")
                continue  # Пропустить точки за камерой

            mp = MapPoint(pts3D[i])
            # Добавляем дескрипторы и наблюдения
            mp.descriptors.append(descriptors1[inlier_matches[i].queryIdx])
            mp.descriptors.append(descriptors2[inlier_matches[i].trainIdx])
            mp.observations.append((idx1, inlier_matches[i].queryIdx))
            mp.observations.append((idx2, inlier_matches[i].trainIdx))
            new_map_points.append(mp)

        self.logger.info(f"Триангулировано {len(new_map_points)} новых точек карты.")
        return new_map_points

    def get_inliers_epipolar(self,
                             keypoints1: List[cv2.KeyPoint],
                             keypoints2: List[cv2.KeyPoint],
                             matches: List[cv2.DMatch]) -> List[cv2.DMatch]:
        """
        Находит инлайерные соответствия на основе эпиполярных ограничений.

        Параметры:
        - keypoints1 (list of cv2.KeyPoint): ключевые точки первого кадра.
        - keypoints2 (list of cv2.KeyPoint): ключевые точки второго кадра.
        - matches (list of cv2.DMatch): список соответствий.

        Возвращает:
        - inlier_matches (list of cv2.DMatch): список инлайерных соответствий.
        """
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

        self.logger.info(f"Found {len(inlier_matches)} inlier matches out of {len(matches)} total matches.")
        return inlier_matches

    def triangulate_new_points(self,
                               keyframe1: Tuple[int, List[cv2.KeyPoint], np.ndarray, np.ndarray],
                               keyframe2: Tuple[int, List[cv2.KeyPoint], np.ndarray, np.ndarray],
                               inlier_matches: List[cv2.DMatch],
                               map_points: List[MapPoint]) -> List[MapPoint]:
        """
        Триангулирует новые точки карты между двумя кейфреймами, если они еще не были триангулированы.

        Параметры:
        - keyframe1 (tuple): кортеж (индекс кадра, ключевые точки, дескрипторы, поза).
        - keyframe2 (tuple): кортеж (индекс кадра, ключевые точки, дескрипторы, поза).
        - inlier_matches (list of cv2.DMatch): список инлайерных соответствий.
        - map_points (list of MapPoint): существующие точки карты.

        Возвращает:
        - new_map_points (list of MapPoint): список новых точек карты.
        """
        new_map_points = []
        idx1, keypoints1, descriptors1, pose1 = keyframe1
        idx2, keypoints2, descriptors2, pose2 = keyframe2

        # Проекционные матрицы
        proj_matrix1 = self.camera_matrix @ pose1
        proj_matrix2 = self.camera_matrix @ pose2

        for match in inlier_matches:
            kp1_idx = match.queryIdx
            kp2_idx = match.trainIdx

            # Проверяем, были ли точки уже триангулированы
            already_triangulated = False
            for mp in map_points:
                if (idx1, kp1_idx) in mp.observations or (idx2, kp2_idx) in mp.observations:
                    already_triangulated = True
                    break

            if already_triangulated:
                continue

            # Координаты точек
            pt1 = keypoints1[kp1_idx].pt
            pt2 = keypoints2[kp2_idx].pt

            # Триангуляция
            pts4D_hom = cv2.triangulatePoints(
                proj_matrix1, proj_matrix2, np.array([pt1]).T, np.array([pt2]).T)
            pts3D = (pts4D_hom[:3] / pts4D_hom[3]).reshape(3)

            # Создаем новую точку карты
            mp = MapPoint(pts3D)
            mp.descriptors.append(descriptors1[kp1_idx])
            mp.descriptors.append(descriptors2[kp2_idx])
            mp.add_observation(idx1, kp1_idx)
            mp.add_observation(idx2, kp2_idx)
            new_map_points.append(mp)

        return new_map_points

    def clean_local_map(self, map_points: List[MapPoint], current_pose: np.ndarray, max_distance=50.0) -> List[MapPoint]:
        """
        Очищает локальную карту, удаляя точки, которые слишком далеко от текущей позы камеры.

        Параметры:
        - map_points (List[MapPoint]): список точек карты.
        - current_pose (numpy.ndarray): текущая поза камеры.
        - max_distance (float): максимальное расстояние до точки карты, после которого она считается удалённой.

        Возвращает:
        - cleaned_map_points (List[MapPoint]): обновлённый список точек карты.
        """
        cleaned_map_points = []
        R = current_pose[:, :3]
        t = current_pose[:, 3]
        camera_position = -R.T @ t  # Позиция камеры в мировой системе координат

        for mp in map_points:
            distance = np.linalg.norm(mp.coordinates - camera_position)
            if distance < max_distance:
                cleaned_map_points.append(mp)

        return cleaned_map_points

    def update_connections_after_pnp(self, map_points: List[MapPoint], curr_keypoints: List[cv2.KeyPoint], curr_descriptors: np.ndarray, frame_idx: int):
        """
        Обновляет связи точек карты после решения PnP, добавляя новые наблюдения.

        Параметры:
        - map_points (list of MapPoint): список точек карты.
        - curr_keypoints (list of cv2.KeyPoint): ключевые точки текущего кадра.
        - curr_descriptors (np.ndarray): дескрипторы текущего кадра.
        - frame_idx (int): индекс текущего кадра.
        """
        # Сопоставляем дескрипторы точек карты с текущими дескрипторами
        map_descriptors = []
        map_indices = []
        for idx, mp in enumerate(map_points):
            if mp.descriptors:
                map_descriptors.append(mp.descriptors[0])
                map_indices.append(idx)

        if not map_descriptors:
            return

        map_descriptors = np.array(map_descriptors, dtype=np.uint8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(map_descriptors, curr_descriptors)

        # Обновляем наблюдения точек карты
        for match in matches:
            mp_idx = map_indices[match.queryIdx]
            kp_idx = match.trainIdx
            map_points[mp_idx].add_observation(frame_idx, kp_idx)
            map_points[mp_idx].matched_times += 1
