import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics_logger")

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
        self.image_width = image_width
        self.image_height = image_height
        self.camera_matrix = config.CAMERA_MATRIX
        self.dist_coeffs = np.zeros((4, 1)) 
        self.logger = logging.getLogger(__name__)
        self.logger.info("OdometryCalculator инициализирован с приблизительной матрицей камеры.")

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
            threshold=config.E_RANSAC_THRESHOLD
        )

        if E is None:
            self.logger.warning("Не удалось вычислить матрицу Essential.")
            return None
        
        src_pts_inliers = src_pts[mask.ravel() == 1]
        dst_pts_inliers = dst_pts[mask.ravel() == 1]

        # Вычисляем ошибку переноса
        error = self.calculate_symmetric_transfer_error(E, src_pts_inliers, dst_pts_inliers)
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
        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=config.H_RANSAC_THRESHOLD)
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
        Использует матрицу Essential для восстановления относительного движения камеры с помощью recoverPose.
        """
        # Определяем соответствующие точки
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        # Используем recoverPose для восстановления вращения и трансляции
        _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix, mask=None)
        self.logger.info(f"Восстановлены R: {R}, t: {t} с помощью recoverPose.")

        # Логируем маску инлайеров
        self.logger.info(f"Маска инлайеров: {mask_pose.sum()} из {len(mask_pose)} точек.")

        # Вычисляем эпиполярные линии для симметричной ошибки переноса
        F = np.linalg.inv(self.camera_matrix).T @ E @ np.linalg.inv(self.camera_matrix)  # Восстановление F
        lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

        # Расчет ошибок для обеих сторон
        error1 = np.abs(np.sum(src_pts * lines1[:, :2], axis=1) + lines1[:, 2]) / np.linalg.norm(lines1[:, :2], axis=1)
        error2 = np.abs(np.sum(dst_pts * lines2[:, :2], axis=1) + lines2[:, 2]) / np.linalg.norm(lines2[:, :2], axis=1)

        # Возвращаем среднюю симметричную ошибку
        error = np.mean(error1 + error2)
        self.logger.info(f"Симметричная ошибка переноса для Essential: {error}")

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
        """
        if len(map_points) == 0:
            self.logger.warning("No map_points available.")
            return [], []
        
        visible_map_points = []
        projected_points = []

        # Разбираем текущую позу камеры
        R_curr = curr_pose[:, :3]
        t_curr = curr_pose[:, 3].reshape(3, 1)

        # Определяем границы изображения
        # Используем реальные размеры изображения, переданные при инициализации
        image_width = self.image_width
        image_height = self.image_height

        self.logger.info(f"Total map points: {len(map_points)}")

        # Оптимизация: используем векторизацию для преобразования всех точек
        map_coords = np.array([mp.coordinates for mp in map_points]).T  # 3xN
        points_cam = R_curr @ map_coords + t_curr  # 3xN

        # Фильтрация по глубине
        valid_depth = points_cam[2, :] > 0
        points_cam = points_cam[:, valid_depth]
        filtered_map_points = [mp for mp, valid in zip(map_points, valid_depth) if valid]

        # Проекция на плоскость изображения
        points_proj = self.camera_matrix @ points_cam  # 3xN
        points_proj /= points_proj[2, :]

        x = points_proj[0, :]
        y = points_proj[1, :]

        # Фильтрация по границам изображения
        in_image = (x >= 0) & (x < image_width) & (y >= 0) & (y < image_height)
        x = x[in_image]
        y = y[in_image]
        final_map_points = [mp for mp, valid in zip(filtered_map_points, in_image) if valid]

        visible_map_points = final_map_points
        projected_points = list(zip(x, y))

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

        # Пакетная обработка для ускорения
        distances, indices = tree.query(projected_points, distance_upper_bound=radius)
        valid_matches = distances != float('inf')

        for mp, idx, dist, valid in zip(visible_map_points, indices, distances, valid_matches):
            if valid:
                keypoint_idx = idx
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

        if len(inlier_matches) < 8:
            self.logger.warning("Недостаточно совпадений для триангуляции новых map points.")
            return []

        # Извлекаем 2D-точки из совпадений
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in inlier_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in inlier_matches])

        # Извлекаем матрицы вращения и векторы трансляции
        R1, t1 = pose1[:, :3], pose1[:, 3].reshape(3, 1)
        R2, t2 = pose2[:, :3], pose2[:, 3].reshape(3, 1)

        # Создаем проекционные матрицы для обоих кадров
        proj_matrix1 = self.camera_matrix @ np.hstack((R1, t1))
        proj_matrix2 = self.camera_matrix @ np.hstack((R2, t2))

        # Триангулируем 3D-точки
        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D_hom = pts4D_hom / pts4D_hom[3]  # Нормализация однородных координат
        pts3D = pts3D_hom[:3].T  # Приводим к форме (N, 3)

        # Настраиваем порог ошибки пере-проекции
        reprojection_error_threshold = 2

        new_map_points = []
        for i, point3D in enumerate(pts3D):
            # Отбрасываем точки за камерой
            if point3D[2] <= 0:
                continue

            # Пере-проецируем 3D-точку в оба кадра для оценки ошибки
            point3D_hom = np.append(point3D, 1).reshape(4, 1)
            proj_pt1_hom = proj_matrix1 @ point3D_hom
            proj_pt2_hom = proj_matrix2 @ point3D_hom

            proj_pt1 = (proj_pt1_hom[:2] / proj_pt1_hom[2]).flatten()
            proj_pt2 = (proj_pt2_hom[:2] / proj_pt2_hom[2]).flatten()

            # Вычисляем ошибки пере-проекции в обоих кадрах
            error1 = np.linalg.norm(proj_pt1 - src_pts[i])
            error2 = np.linalg.norm(proj_pt2 - dst_pts[i])

            # Фильтруем точки с большой ошибкой пере-проекции
            if error1 > reprojection_error_threshold or error2 > reprojection_error_threshold:
                continue

            # Создаем новую MapPoint и добавляем дескрипторы и наблюдения
            mp = MapPoint(point3D)
            mp.descriptors.append(descriptors1[inlier_matches[i].queryIdx])
            mp.descriptors.append(descriptors2[inlier_matches[i].trainIdx])
            mp.observations.append((idx1, inlier_matches[i].queryIdx))
            mp.observations.append((idx2, inlier_matches[i].trainIdx))
            new_map_points.append(mp)

        self.logger.info(f"Триангулировано {len(new_map_points)} новых точек карты после фильтрации.")
        return new_map_points

    def get_inliers_epipolar(self,
                            keypoints1: List[cv2.KeyPoint],
                            keypoints2: List[cv2.KeyPoint],
                            matches: List[cv2.DMatch],
                            epipolar_threshold: float = 0.01) -> List[cv2.DMatch]:
        """
        Находит инлайерные соответствия на основе эпиполярных ограничений и эпиполярной ошибки.

        Параметры:
        - keypoints1 (list of cv2.KeyPoint): ключевые точки первого кадра.
        - keypoints2 (list of cv2.KeyPoint): ключевые точки второго кадра.
        - matches (list of cv2.DMatch): список соответствий.
        - epipolar_threshold (float): порог эпиполярного расстояния для фильтрации.

        Возвращает:
        - inlier_matches (list of cv2.DMatch): список инлайерных соответствий.
        """
        if not matches:
            self.logger.warning("Нет соответствий для обработки.")
            return []

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Вычисляем фундаментальную матрицу с помощью RANSAC
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

        if F is None or mask is None:
            self.logger.warning("Не удалось найти фундаментальную матрицу или маску.")
            return []

        # Нормализация фундаментальной матрицы
        F = F / np.linalg.norm(F)

        # Фильтрация инлайеров на основе RANSAC
        initial_inliers = [m for i, m in enumerate(matches) if mask[i]]

        # Дополнительная фильтрация с использованием эпиполярной ошибки
        inlier_matches = []
        for match in initial_inliers:
            pt1 = np.array(keypoints1[match.queryIdx].pt, dtype=np.float32).reshape(1, 2)
            pt2 = np.array(keypoints2[match.trainIdx].pt, dtype=np.float32).reshape(1, 2)

            # Вычисляем эпиполярные линии для обеих точек
            line1 = cv2.computeCorrespondEpilines(pt2.reshape(-1,1,2), 2, F).reshape(-1,3)[0]
            line2 = cv2.computeCorrespondEpilines(pt1.reshape(-1,1,2), 1, F).reshape(-1,3)[0]

            # Эпиполярное расстояние для каждой точки
            error1 = abs(line1[0]*pt1[0,0] + line1[1]*pt1[0,1] + line1[2]) / np.sqrt(line1[0]**2 + line1[1]**2)
            error2 = abs(line2[0]*pt2[0,0] + line2[1]*pt2[0,1] + line2[2]) / np.sqrt(line2[0]**2 + line2[1]**2)

            if (error1 + error2) < epipolar_threshold:
                inlier_matches.append(match)

        self.logger.info(f"Найдено {len(inlier_matches)} инлайерных соответствий из {len(matches)} исходных.")
        return inlier_matches

    def clean_local_map(self, map_points: List[MapPoint], current_pose: np.ndarray) -> List[MapPoint]:
        """
        Очищает локальную карту, удаляя точки, которые слишком далеко от текущей позы камеры.

        Параметры:
        - map_points (List[MapPoint]): список точек карты.
        - current_pose (numpy.ndarray): текущая поза камеры.

        Возвращает:
        - cleaned_map_points (List[MapPoint]): обновлённый список точек карты.
        """
        cleaned_map_points = []
        R = current_pose[:, :3]
        t = current_pose[:, 3]
        camera_position = -R.T @ t  # Позиция камеры в мировой системе координат

        for mp in map_points:
            distance = np.linalg.norm(mp.coordinates - camera_position)
            if distance < config.MAP_CLEAN_MAX_DISTANCE:
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
        # Сбор дескрипторов для сопоставления
        map_descriptors = []
        map_indices = []
        for idx, mp in enumerate(map_points):
            # Используем последний дескриптор для большей актуальности наблюдений
            if mp.descriptors:
                map_descriptors.append(mp.descriptors[-1])
                map_indices.append(idx)

        if not map_descriptors:
            return

        map_descriptors = np.array(map_descriptors, dtype=np.uint8)

        # Инициализируем BFMatcher и выполняем поиск K ближайших соседей
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(map_descriptors, curr_descriptors, k=2)

        # Применяем тест отношения расстояний для фильтрации надёжных совпадений
        good_matches = []
        ratio_thresh = 0.75  # Порог отношения (можно настроить)
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # Обновляем наблюдения точек карты на основе отфильтрованных соответствий
        for match in good_matches:
            mp_idx = map_indices[match.queryIdx]
            kp_idx = match.trainIdx
            map_points[mp_idx].add_observation(frame_idx, kp_idx)
            map_points[mp_idx].matched_times += 1

