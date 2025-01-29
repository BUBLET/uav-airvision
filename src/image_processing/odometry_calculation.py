import cv2
import numpy as np
import logging

from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from .map_point import MapPoint


class OdometryCalculator:
    def __init__(self, 
                 image_width: int, 
                 image_height: int, 
                 camera_matrix: Optional[np.ndarray] = None,
                 e_ransac_threshold: float = 1.0,
                 h_ransac_threshold: float = 3.0,
                 distance_threshold: int = 30,
                 map_clean_max_distance: float = 100.0,
                 reprojection_threshold: float = 8.0,
                 ratio_thresh: float = 0.75,
                 dist_coeffs: Optional[np.ndarray] = None,
                 logger: Optional[logging.Logger] = None,
                 ckd_radius: int = 5
        ):

        self.logger = logger or logging.getLogger(__name__)
        self.image_width = image_width
        self.image_height = image_height

        self.E_RANSAC_THRESHOLD = e_ransac_threshold
        self.H_RANSAC_THRESHOLD = h_ransac_threshold
        self.DISTANCE_THRESHOLD = distance_threshold
        self.MAP_CLEAN_MAX_DISTANCE = map_clean_max_distance
        self.REPROJECTION_THRESHOLD = reprojection_threshold
        self.RATIO_THRESH = ratio_thresh
        self.CKD_RADIUS = ckd_radius

        if camera_matrix is None:
            self.camera_matrix = np.array([[800, 0, image_width / 2],
                                           [0, 800, image_height / 2],
                                           [0, 0, 1]
                                           ], dtype=np.float64)
            self.logger.info("Используется матрица камеры по умолчанию")
        else:
            self.camera_matrix = camera_matrix
            self.logger.info("Используется переданная матрица камеры")

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1)) 
        else:
            self.dist_coeffs = dist_coeffs
    
        self.logger.info("OdometryCalculator инициализирован")

    def triangulate_new_map_points(
        self,
        keyframe1: Tuple[int, List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray],
        keyframe2: Tuple[int, List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray],
        matches: List[cv2.DMatch]
    ) -> List[MapPoint]:
        """
        Триангулирует новые map points между двумя keyframes.

        """
        idx1, keypoints1, descriptors1, pose1 = keyframe1
        idx2, keypoints2, descriptors2, pose2 = keyframe2

        if descriptors1 is None or descriptors2 is None:
            self.logger.warning("triangulate_new_map_points: один из дескрипторов None.")
            return []

        if len(descriptors1) == 0 or len(descriptors2) == 0:
            self.logger.warning("triangulate_new_map_points: один из дескрипторных массивов пуст.")
            return []

        if len(matches) < 8:
            self.logger.warning("Недостаточно совпадений для триангуляции новых map points (<8).")
            return []

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        R1, t1 = pose1[:, :3], pose1[:, 3]
        R2, t2 = pose2[:, :3], pose2[:, 3]

        proj_matrix1 = self.camera_matrix @ np.hstack((R1, t1.reshape(3, 1)))
        proj_matrix2 = self.camera_matrix @ np.hstack((R2, t2.reshape(3, 1)))

        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = (pts4D_hom[:3] / pts4D_hom[3]).T  # (N, 3)

        new_map_points = []
        for i in range(pts3D.shape[0]):
            mp = MapPoint(coordinates=pts3D[i])

            mp.descriptors.append(descriptors1[matches[i].queryIdx])
            mp.descriptors.append(descriptors2[matches[i].trainIdx])

            mp.observations.append((idx1, matches[i].queryIdx))
            mp.observations.append((idx2, matches[i].trainIdx))

            new_map_points.append(mp)

        return new_map_points

    def _calculate_symmetric_transfer_error_homography(
            self,
            H: np.ndarray,
            src_pts: np.ndarray,
            dst_pts: np.ndarray
    ) -> float:
        
        src_pts_h = cv2.convertPointsToHomogeneous(src_pts)[:, 0, :]
        dst_pts_h = cv2.convertPointsToHomogeneous(dst_pts)[:, 0, :]

        src_to_dst = (H @ src_pts_h.T).T
        dst_to_src = (np.linalg.inv(H) @ dst_pts_h.T).T

        # Нормируем
        src_to_dst /= src_to_dst[:, 2][:, np.newaxis]
        dst_to_src /= dst_to_src[:, 2][:, np.newaxis]

        error1 = np.linalg.norm(dst_pts - src_to_dst[:, :2], axis=1)
        error2 = np.linalg.norm(src_pts - dst_to_src[:, :2], axis=1)
        return float(np.mean(error1 + error2))

    def _calculate_symmetric_transfer_error_essential(
            self,
            E: np.ndarray,
            src_pts: np.ndarray,
            dst_pts: np.ndarray
    ) -> float:
        camera_inv = np.linalg.inv(self.camera_matrix)
        F = camera_inv.T @ E @ camera_inv

        lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

        error1 = np.abs(np.sum(src_pts * lines1[:, :2], axis=1) + lines1[:, 2]) / np.linalg.norm(lines1[:, :2], axis=1)
        error2 = np.abs(np.sum(dst_pts * lines2[:, :2], axis=1) + lines2[:, 2]) / np.linalg.norm(lines2[:, :2], axis=1)
        return float(np.mean(error1 + error2))

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
            return self._calculate_symmetric_transfer_error_homography(matrix, src_pts, dst_pts)
        else:
            return self._calculate_symmetric_transfer_error_essential(matrix, src_pts, dst_pts)

    def _check_essential_inputs(
            self,
            matches: List[cv2.DMatch]
    ) -> bool:
        """ Проверяет достаточно ли совпадений для вычисления Е """
        
        if not matches:
            self.logger.warning("Нет совпадения для вычисления E")
            return False
        
        return True

    def _extract_corresponding_points(
            self,
            prev_keypoints: List[cv2.KeyPoint],
            curr_keypoints: List[cv2.KeyPoint],
            matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Извлекает 2D точки из списка соответствий """

        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])
        
        return src_pts, dst_pts
        
    def _find_essential_mat(
            self,
            src_pts: np.ndarray,
            dst_pts: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ 
        Обертка cv2.findEssentialMat (RANSAC)

        Возвращает (E, mask) или None
        """

        E, mask = cv2.findEssentialMat(
            src_pts,
            dst_pts,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.E_RANSAC_THRESHOLD
        )
        if E is None:
            self.logger.warning("Не удалось вычислить Essential")
            return None
        
        return E, mask

    def calculate_essential_matrix(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Вычисляет матрицу Essential между предыдущими и текущими кадрами на основе ключевых точек.

        Возвращает:
        - (E, mask, error) или None.
        """

        if not self._check_essential_inputs(matches):
            return None  

        src_pts, dst_pts = self._extract_corresponding_points(prev_keypoints, curr_keypoints, matches)

        result = self._find_essential_mat(src_pts, dst_pts)
        if result is None:
            return None  

        E, mask = result

        inliers_count = np.count_nonzero(mask)
        self.logger.debug(f"[E] Inliers = {inliers_count}/{len(matches)}")

        src_inliers = src_pts[mask.ravel() == 1]
        dst_inliers = dst_pts[mask.ravel() == 1]
        error = self.calculate_symmetric_transfer_error(E, src_inliers, dst_inliers, is_homography=False)

        return E, mask, error

    def _check_decompose_essential_inputs(
            self,
            E: np.ndarray,
            matches: List[cv2.DMatch]
    ) -> bool:
        """ Проверяет корректна ли Е и достаточно ли совпадений """
        
        if E is None or E.shape != (3, 3):
            self.logger.warning("Некорректная Е")
            return False
        
        if len(matches) < 5:
            self.logger.warning("Недостаточно совпадений (<5)")
            return False
        
        return True

    def _recover_pose(
            self,
            E: np.ndarray,
            src_pts: np.ndarray,
            dst_pts: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """ Обертка для cv2.recoverPose """
        
        retval, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)
        if retval < 1:
            self.logger.warning("recoverPose не смог восстановить достаточное число точек.")
            return None, None, None

        self.logger.info(f"E->R,t: inliers={mask_pose.sum()}/{len(mask_pose)}")
        return R, t, mask_pose

    def decompose_essential(
            self,
            E: np.ndarray,
            prev_keypoints: List[cv2.KeyPoint],
            curr_keypoints: List[cv2.KeyPoint],
            matches: List[cv2.DMatch]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Декомпозирует Е с помощью recoverPose.

        Возвращает 
        ----------
        R, t, mask_pose
        """
        if not self._check_decompose_essential_inputs(E, matches):
            return None, None, None
        
        src_pts, dst_pts = self._extract_corresponding_points(prev_keypoints, curr_keypoints, matches)
        
        return self._recover_pose(E, src_pts, dst_pts)

    def _check_homography_inputs(
        self,
        H: np.ndarray,
        matches: List[cv2.DMatch]
    ) -> bool:
        """Проверяет корректность H и достаточно ли совпадений"""
        
        if H is None or H.shape != (3, 3):
            self.logger.warning("Некорректная матрица H")
            return False
        
        if len(matches) < 4:
            self.logger.warning("Недостаточно совпадений для декомпозиции H(<4).")
            return False
        
        return True

    def _find_homography_mat(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Обёртка над cv2.findHomography (RANSAC).
        Возвращает (H, mask) или None
        """
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.H_RANSAC_THRESHOLD
        )

        if H is None:
            self.logger.warning("Не удалось вычислить матрицу Homography.")
            return None
        
        return H, mask

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
        - H (numpy.ndarray)
        - mask (numpy.ndarray)
        - error (float)

        Или None если H не удалось вычислить.
        """
        if len(matches) < 4:
            self.logger.warning("Недостаточно совпадений для вычисления Homography (<4).")
            return None
    
        src_pts, dst_pts = self._extract_corresponding_points(prev_keypoints, curr_keypoints, matches)

        result = self._find_homography_mat(src_pts, dst_pts)
        if result is None:
            return None  

        H, mask = result

        inliers_count = np.count_nonzero(mask)
        self.logger.debug(f"[H] Inliers = {inliers_count}/{len(matches)}")

        error = self.calculate_symmetric_transfer_error(H, src_pts, dst_pts, is_homography=True)
        return H, mask, error

    def _decompose_homography_mat(
        self,
        H: np.ndarray
    ) -> Optional[Tuple[int, List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        """
        Обёртка для cv2.decomposeHomographyMat.

        Возвращает (retval, Rs, ts, normals) или None.

        """
        retval, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)
        
        if retval == 0:
            self.logger.warning("Не удалось декомпозировать H")
            return None
        
        return (retval, Rs, ts, normals)

    def _count_inliers_for_homography_solution(
        self,
        R_candidate: np.ndarray,
        t_candidate: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Триангулирует точки для (R_candidate, t_candidate) и считает,
        сколько из них имеют глубину z>0.
        Возвращает (num_inliers, mask_depth).
        """
        proj_matrix1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = self.camera_matrix @ np.hstack((R_candidate, t_candidate.reshape(3, 1)))

        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = (pts4D_hom[:3] / pts4D_hom[3]).T  # N x 3

        depths = pts3D[:, 2]
        mask_depth = depths > 0
        num_inliers = np.sum(mask_depth)
        return num_inliers, mask_depth

    def decompose_homography(
        self,
        H: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Декомпозирует Homography, выбирая лучшее решение по числу точек перед камерой.
        Возвращает (best_R, best_t, best_mask, best_num_inliers).
        """
        if not self._check_homography_inputs(H, matches):
            return None, None, None, 0

        src_pts, dst_pts = self._extract_corresponding_points(prev_keypoints, curr_keypoints, matches)
        result = self._decompose_homography_mat(H)
        if result is None:
            return None, None, None, 0
        retval, Rs, ts, normals = result

        best_num_inliers = -1
        best_R, best_t, best_mask = None, None, None

        for R_candidate, t_candidate, normal in zip(Rs, ts, normals):
            num_inliers, mask_depth = self._count_inliers_for_homography_solution(
                R_candidate, t_candidate, src_pts, dst_pts
            )
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_R = R_candidate
                best_t = t_candidate
                best_mask = mask_depth

        if best_num_inliers < 1:
            self.logger.warning("Не удалось найти решение с положительной глубиной точек.")
            return None, None, None, 0

        return best_R, best_t, best_mask, best_num_inliers

    def _filter_inlier_matches(
        self,
        matches: List[cv2.DMatch],
        mask_pose: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Возвращает список совпадений, у которых mask_pose[i] == True.
        """
        return [m for i, m in enumerate(matches) if mask_pose[i]]

    def _extract_corresponding_points_for_inliers(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Аналог _extract_corresponding_points, но для заранее отфильтрованных inlier_matches.
        """
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in inlier_matches])
        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in inlier_matches])
        return src_pts, dst_pts

    def _triangulate(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Собственно триангуляция точек (обёртка над cv2.triangulatePoints).

        Возвращает pts3D (N x 3).
        """
        proj_matrix1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = self.camera_matrix @ np.hstack((R, t))

        pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, src_pts.T, dst_pts.T)
        pts3D = (pts4D_hom[:3] / pts4D_hom[3]).T
        return pts3D

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
        Триангулирует 3D-точки, используя инлайерные соответствия из mask_pose.
        Возвращает (pts3D, inlier_matches).
        """
        inlier_matches = self._filter_inlier_matches(matches, mask_pose)
        if not inlier_matches:
            self.logger.warning("Нет соответствий для триангуляции.")
            return np.empty((0, 3)), []

        src_pts, dst_pts = self._extract_corresponding_points_for_inliers(prev_keypoints, curr_keypoints, inlier_matches)
        pts3D = self._triangulate(src_pts, dst_pts, R, t)
        return pts3D, inlier_matches

    def _find_fundamental_matrix(
            self,
            src_pts: np.ndarray,
            dst_pts: np.ndarray,
            ransac_thresh: float,
            confidence: float = 0.99
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Находит фундаментальную матрицу между двумя наборами точек.

        """
        if len(src_pts) < 8:
            self.logger.warning("Недостаточно точек для вычисления фундаментальной матрицы.")
            return None, None

        F, mask = cv2.findFundamentalMat(
            src_pts,
            dst_pts,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=confidence
        )
        if F is None or mask is None:
            self.logger.warning("Не удалось вычислить фундаментальную матрицу.")
            return None, None

        return F, mask
    
    def get_inliers_epipolar(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        ransac_thresh: float = 1.0
    ) -> List[cv2.DMatch]:
        """
        Главный метод для получения инлайнеров через эпиполярную фильтрацию.

        """
        src_pts, dst_pts = self._extract_corresponding_points(prev_keypoints, curr_keypoints, matches)

        F, mask = self._find_fundamental_matrix(src_pts, dst_pts, ransac_thresh)
        
        if mask is not None:
            inlier_matches = self._filter_inlier_matches(matches, mask)
        else:
            return []

        self.logger.info(
            f"get_inliers_epipolar: {len(inlier_matches)} / {len(matches)} inliers"
        )
        return inlier_matches

    def check_triangulation_angle(
        self,
        R: np.ndarray,
        t: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        mask_pose: Optional[np.ndarray] = None
    ) -> float:
        """
        Вычисляет параллакс между двумя кадрами,

        """
        # 1) Триангулируем 3D-точки, используя готовый метод
        pts3D, inlier_matches = self.triangulate_points(
            R, t,
            prev_keypoints,
            curr_keypoints,
            matches,
            mask_pose
        )
        if len(pts3D) == 0:
            self.logger.warning("check_triangulation_angle: Нет 3D-точек для вычисления угла.")
            return 0.0

        rays_cam1 = pts3D

        rays_cam2 = (R @ rays_cam1.T + t.reshape(3, 1)).T  # shape: (N,3)


        norms1 = np.linalg.norm(rays_cam1, axis=1, keepdims=True) + 1e-9
        norms2 = np.linalg.norm(rays_cam2, axis=1, keepdims=True) + 1e-9

        rays_cam1_norm = rays_cam1 / norms1
        rays_cam2_norm = rays_cam2 / norms2

        cos_angles = np.sum(rays_cam1_norm * rays_cam2_norm, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        angles = np.arccos(cos_angles) 

        median_angle = np.median(angles)
        return float(median_angle)

    def _filter_points_by_depth(
        self,
        pts3D: np.ndarray,
        inlier_matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, List[cv2.DMatch]]:
        """
        Возвращает (pts3D_valid, inlier_matches_valid) только для тех точек, где Z > 0.
        """
        valid_mask = pts3D[:, 2] > 0
        pts3D_valid = pts3D[valid_mask]
        inlier_matches_valid = [m for m, v in zip(inlier_matches, valid_mask) if v]
        return pts3D_valid, inlier_matches_valid

    def _build_map_points(
        self,
        pts3D_valid: np.ndarray,
        descriptors_prev: np.ndarray,
        descriptors_curr: np.ndarray,
        inlier_matches_valid: List[cv2.DMatch],
        prev_frame_idx: int,
        curr_frame_idx: int,
    ) -> List[MapPoint]:
        """
        Создаёт объекты MapPoint из отсортированных 3D-точек и дескрипторов.
        """
        map_points = []
        for i, point_coords in enumerate(pts3D_valid):
            mp = MapPoint(coordinates=point_coords)
            mp.descriptors.append(descriptors_prev[i])
            mp.descriptors.append(descriptors_curr[i])
            mp.observations.append((prev_frame_idx, inlier_matches_valid[i].queryIdx))
            mp.observations.append((curr_frame_idx, inlier_matches_valid[i].trainIdx))
            map_points.append(mp)
        return map_points

    def convert_points_to_structure(
        self,
        pts3D: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint],
        curr_keypoints: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch],
        prev_descriptors: np.ndarray,
        curr_descriptors: np.ndarray,
        prev_frame_idx: int,
        curr_frame_idx: int,
    ) -> List[MapPoint]:
        """
        Преобразует триангулированные 3D-точки в объекты MapPoint, связывает их с дескрипторами.
        """
        if len(pts3D) != len(inlier_matches):
            self.logger.warning("Количество 3D-точек не совпадает с количеством inlier_matches.")
            return []

        # Фильтруем по глубине
        pts3D_valid, inlier_matches_valid = self._filter_points_by_depth(pts3D, inlier_matches)
        if len(pts3D_valid) == 0:
            self.logger.warning("Нет 3D-точек с положительной глубиной.")
            return []

        # Собираем дескрипторы
        query_idxs = [m.queryIdx for m in inlier_matches_valid]
        train_idxs = [m.trainIdx for m in inlier_matches_valid]
        descriptors_prev = prev_descriptors[query_idxs]
        descriptors_curr = curr_descriptors[train_idxs]

        # Формируем объекты MapPoint
        map_points = self._build_map_points(
            pts3D_valid,
            descriptors_prev,
            descriptors_curr,
            inlier_matches_valid,
            prev_frame_idx,
            curr_frame_idx,
        )
        self.logger.info(f"Добавлено {len(map_points)} новых точек карты.")
        return map_points

    def _filter_points_by_depth_and_image_bounds(
        self,
        map_points: List[MapPoint],
        curr_pose: np.ndarray
    ) -> Tuple[List[MapPoint], np.ndarray, np.ndarray]:
        """
        Фильтрует map_points по глубине (Z>0) и границам изображения, возвращает
        (filtered_map_points, x, y) — координаты проекций на изображение.
        """
        R_curr = curr_pose[:, :3]
        t_curr = curr_pose[:, 3].reshape(3, 1)

        map_coords = np.array([mp.coordinates for mp in map_points]).T  # (3, N)
        points_cam = R_curr @ map_coords + t_curr  # (3, N)

        valid_depth = points_cam[2, :] > 0
        points_cam = points_cam[:, valid_depth]
        filtered_points = [mp for mp, v in zip(map_points, valid_depth) if v]

        proj_points = self.camera_matrix @ points_cam
        proj_points /= proj_points[2, :]

        x = proj_points[0, :]
        y = proj_points[1, :]

        in_image = (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height)
        x = x[in_image]
        y = y[in_image]
        final_map_points = [mp for mp, v in zip(filtered_points, in_image) if v]

        return final_map_points, x, y

    def _match_projected_points_with_keypoints(
        self,
        visible_mp: List[MapPoint],
        x: np.ndarray,
        y: np.ndarray,
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray
    ) -> Tuple[List[MapPoint], List[int]]:
        """
        Сопоставляет 2D-проекции видимых map points (x,y) с текущими ключевыми точками,
        возвращает списки (matched_map_points, matched_keypoint_indices).
        """
        if len(visible_mp) == 0:
            return [], []

        keypoints_coords = np.array([kp.pt for kp in curr_keypoints])
        tree = cKDTree(keypoints_coords)

        # Поиск ближайших ключевых точек в радиусе CKD_RADIUS
        query_points = list(zip(x, y))
        distances, indices = tree.query(query_points, distance_upper_bound=self.CKD_RADIUS)
        valid_matches = distances != float('inf')

        matched_map_points = []
        matched_keypoint_indices = []

        for mp, idx, dist, valid in zip(visible_mp, indices, distances, valid_matches):
            if valid and mp.descriptors:
                # Берем последний дескриптор map point
                mp_descriptor = mp.descriptors[-1]
                kp_descriptor = curr_descriptors[idx]
                distance = cv2.norm(mp_descriptor, kp_descriptor, cv2.NORM_HAMMING)
                if distance < self.DISTANCE_THRESHOLD:
                    matched_map_points.append(mp)
                    matched_keypoint_indices.append(idx)

        return matched_map_points, matched_keypoint_indices

    def visible_map_points(
        self,
        map_points: List[MapPoint],
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray,
        curr_pose: np.ndarray
    ) -> Tuple[List[MapPoint], List[int]]:
        """
        Определяет видимые точки карты в текущем кадре и сопоставляет их с ключевыми точками.
        Возвращает (matched_map_points, matched_keypoint_indices).
        """
        if not map_points:
            self.logger.warning("Список map_points пуст.")
            return [], []

        # 1) Преобразуем в координаты текущей камеры
        visible_mp, x, y = self._filter_points_by_depth_and_image_bounds(map_points, curr_pose)
        if not visible_mp:
            self.logger.warning("Нет видимых точек после фильтрации по глубине и границам.")
            return [], []

        # 2) Сопоставляем с ключевыми точками
        matched_map_points, matched_keypoint_indices = self._match_projected_points_with_keypoints(
            visible_mp, x, y, curr_keypoints, curr_descriptors
        )
        return matched_map_points, matched_keypoint_indices

    def clean_local_map(
        self,
        map_points: List[MapPoint],
        current_pose: np.ndarray
    ) -> List[MapPoint]:
        """
        Удаляет точки, которые слишком далеки от текущей позы камеры.
        Возвращает обновлённый список map points.
        """
        R = current_pose[:, :3]
        t = current_pose[:, 3]
        camera_position = -R.T @ t

        cleaned_map_points = []
        for mp in map_points:
            dist = np.linalg.norm(mp.coordinates - camera_position)
            if dist < self.MAP_CLEAN_MAX_DISTANCE:
                cleaned_map_points.append(mp)

        self.logger.info(f"Очистка карты: осталось {len(cleaned_map_points)} из {len(map_points)}.")
        return cleaned_map_points

    def _filter_map_points_with_descriptors(
        self,
        map_points: List[MapPoint]
    ) -> List[MapPoint]:
        """
        Оставляет только те map points, у которых есть хотя бы один дескриптор.
        """
        return [mp for mp in map_points if mp.descriptors]

    def _knn_match(
        self,
        map_descriptors: np.ndarray,
        curr_descriptors: np.ndarray
    ) -> List[List[cv2.DMatch]]:
        """
        Выполняет KNN match между дескрипторами карты и дескрипторами текущего кадра.
        """
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(map_descriptors, curr_descriptors, k=2)
        return knn_matches

    def _lowe_ratio_test(
        self,
        knn_matches: List[List[cv2.DMatch]],
        ratio_thresh: float
    ) -> List[cv2.DMatch]:
        """
        Применяет Lowe ratio test к результатам KNN, возвращая "хорошие" совпадения.
        """
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    def _update_map_points_observations(
        self,
        valid_map_points: List[MapPoint],
        good_matches: List[cv2.DMatch],
        map_indices: List[int],
        frame_idx: int
    ) -> None:
        """
        Обновляет наблюдения map points на основе списка "хороших" совпадений.
        """
        for gm in good_matches:
            mp_idx = map_indices[gm.queryIdx]
            kp_idx = gm.trainIdx
            valid_map_points[mp_idx].add_observation(frame_idx, kp_idx)
            valid_map_points[mp_idx].matched_times += 1

    def update_connections_after_pnp(
        self,
        map_points: List[MapPoint],
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray,
        frame_idx: int
    ) -> None:
        """
        Обновляет наблюдения map points на основе текущего кадра (PnP-позы).
        Использует BFMatcher + KNN для сопоставления.
        """
        valid_map_points = self._filter_map_points_with_descriptors(map_points)
        if not valid_map_points:
            return

        map_descriptors = np.array([mp.descriptors[-1] for mp in valid_map_points], dtype=np.uint8)
        map_indices = list(range(len(valid_map_points)))

        knn_matches = self._knn_match(map_descriptors, curr_descriptors)
        good_matches = self._lowe_ratio_test(knn_matches, self.RATIO_THRESH)

        self._update_map_points_observations(valid_map_points, good_matches, map_indices, frame_idx)
