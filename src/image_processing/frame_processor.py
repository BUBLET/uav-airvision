import logging
import numpy as np
import cv2
from typing import List, Optional, Tuple
from src.optimization.ba import BundleAdjustment


class FrameProcessor:
    def __init__(
            self,
            feature_extractor,
            feature_matcher,
            odometry_calculator,
            translation_threshold,
            rotation_threshold,
            triangulation_threshold,
            bundle_adjustment_frames,
            force_keyframe_interval=1,
            homography_inlier_ratio=0.6
    ):
        self.logger = logging.getLogger(__name__)

        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.odometry_calculator = odometry_calculator

        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.triangulation_threshold = triangulation_threshold
        self.bundle_adjustment_frames = bundle_adjustment_frames
        self.force_keyframe_interval = force_keyframe_interval
        self.homography_inlier_ratio = homography_inlier_ratio
        
    def _initialize_map(
           self,
           ref_keypoints: List[cv2.KeyPoint],
           ref_descriptors: np.ndarray, 
           curr_keypoints: List[cv2.KeyPoint],
           curr_descriptors: np.ndarray 
    ):
        matches = self.feature_matcher.match_features(ref_descriptors, curr_descriptors)
        if len(matches) < 4:
            self.logger.warning("Слишком мало совпадений для инициализации.")
            return None

        E_result = self.odometry_calculator.calculate_essential_matrix(ref_keypoints, curr_keypoints, matches)
        H_result = self.odometry_calculator.calculate_homography_matrix(ref_keypoints, curr_keypoints, matches)
        if not E_result or not H_result:
            self.logger.warning("Не удалось вычислить E или H при инициализации.")
            return None

        E, _, error_E = E_result
        H, _, error_H = H_result

        total_error = error_E + error_H
        if total_error == 0:
            self.logger.warning("Сумма ошибок E+H = 0.")
            return None

        # Сравниваем ошибки для выбора E или H
        H_ratio = error_H / total_error
        use_homography = (H_ratio > self.homography_inlier_ratio)

        if use_homography:
            self.logger.info("Используем H для инициализации.")
            R, t, mask_pose = self.odometry_calculator.decompose_homography(
                H, ref_keypoints, curr_keypoints, matches
            )
        else:
            self.logger.info("Используем E для инициализации.")
            R, t, mask_pose = self.odometry_calculator.decompose_essential(
                E, ref_keypoints, curr_keypoints, matches
            )

        if R is None or t is None:
            self.logger.warning("Не удалось восстановить позу из E/H.")
            return None

        # Проверяем угол триангуляции
        median_angle = self.odometry_calculator.check_triangulation_angle(
            R, t, ref_keypoints, curr_keypoints, matches
        )
        self.logger.info(f"Медианный угол триангуляции: {np.rad2deg(median_angle):.2f}°")
        if median_angle < self.triangulation_threshold:
            self.logger.warning("Угол триангуляции ниже порога")
            return None

        return R, t, matches, mask_pose
        
    def _triangulate_initial_points(
        self,
        R: np.ndarray,
        t: np.ndarray,
        ref_keypoints,
        curr_keypoints,
        ref_descriptors: np.ndarray,
        curr_descriptors: np.ndarray,
        matches,
        mask_pose: np.ndarray,
        frame_idx: int,
        map_points: list
    ) -> list:
        """
        Триангулирует 3D точки после инициализации и превращает их в структуру map_points.

        """
        map_points_array, inlier_matches = self.odometry_calculator.triangulate_points(
            R, t, ref_keypoints, curr_keypoints, matches, mask_pose
        )
        ref_frame_idx = frame_idx - 1
        curr_frame_idx = frame_idx
        new_points = self.odometry_calculator.convert_points_to_structure(
            map_points_array,
            ref_keypoints,
            curr_keypoints,
            inlier_matches,
            ref_descriptors,
            curr_descriptors,
            ref_frame_idx,
            curr_frame_idx
        )
        map_points = new_points  
        return map_points
        
    def _estimate_pose(
        self,
        map_points: list,
        curr_keypoints,
        curr_descriptors: np.ndarray,
        last_pose: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Оценивает позу через PnP.

        """
        visible_map_points, map_point_indices = self.odometry_calculator.visible_map_points(
            map_points, curr_keypoints, curr_descriptors, last_pose
        )

        self.logger.info(f"Видимых точек карты: {len(visible_map_points)}")
        if len(visible_map_points) < 4:
            self.logger.warning("Недостаточно видимых точек для PnP.")
            return None

        object_points = np.array([mp.coordinates for mp in visible_map_points], dtype=np.float32)
        image_points = np.array([curr_keypoints[idx].pt for idx in map_point_indices], dtype=np.float32)

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.odometry_calculator.camera_matrix,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        inliers_count = len(inliers) if (inliers is not None and retval) else 0
        self.logger.info(f"PnP инлаеров: {inliers_count}")

        if not retval or inliers_count < 4:
            self.logger.warning("Не удалось восстановить позу из PnP.")
            return None

        R, _ = cv2.Rodrigues(rvec)
        current_pose = np.hstack((R, tvec))
        self.logger.info(f"Текущая поза:\n{current_pose}")
        return current_pose

    def _should_insert_keyframe(
        self,
        last_keyframe_pose: np.ndarray,
        current_pose: np.ndarray,
        frame_index: int
    ) -> bool:
        """
        Логика решения о вставке нового кейфрейма:

        """
        if frame_index % self.force_keyframe_interval == 0:
            self.logger.info(f"Форсированная вставка кейфрейма на кадре {frame_index}")
            return True

        last_hom = np.eye(4)
        last_hom[:3, :4] = last_keyframe_pose
        curr_hom = np.eye(4)
        curr_hom[:3, :4] = current_pose

        delta = np.linalg.inv(last_hom) @ curr_hom
        delta_trans = np.linalg.norm(delta[:3, 3])

        cos_angle = (np.trace(delta[:3, :3]) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        delta_rot = np.arccos(cos_angle)

        if np.isnan(delta_rot):
            self.logger.warning("Невалидное значение поворота при проверке кейфрейма.")
            return False

        self.logger.info(
            f"Кадр {frame_index}, смещение = {delta_trans:.4f}, "
            f"поворот (град) = {np.rad2deg(delta_rot):.2f}"
        )

        if delta_trans > self.translation_threshold or delta_rot > self.rotation_threshold:
            return True
        return False

    def _insert_keyframe(
        self,
        frame_idx: int,
        curr_keypoints,
        curr_descriptors: np.ndarray,
        current_pose: np.ndarray,
        keyframes: list,
        map_points: list
    ):
        """
        Добавляет новый кейфрейм и триангулирует новые точки на основе текущего и предыдущего кейфрейма.

        """
        self.logger.info(f"Добавляем кейфрейм на кадре {frame_idx}.")
        keyframes.append((frame_idx, curr_keypoints, curr_descriptors, current_pose))
        if len(keyframes) < 2:
            return

        prev_keyframe = keyframes[-2]
        matches = self.feature_matcher.match_features(prev_keyframe[2], curr_descriptors)
        inlier_matches = self.odometry_calculator.get_inliers_epipolar(
            prev_keyframe[1], curr_keypoints, matches
        )
        self.logger.info(f"Инлаеров: {len(inlier_matches)}")

        if len(inlier_matches) < 8:
            self.logger.warning("Слишком мало совпадений для триангуляции.")
            return

        new_map_points = self.odometry_calculator.triangulate_new_map_points(
            prev_keyframe,
            keyframes[-1],
            inlier_matches
        )
        map_points.extend(new_map_points)
        self.logger.info(f"Всего точек на карте: {len(map_points)}")

    def _run_local_ba(
            self, 
            recent_keyframes: list, 
            map_points: list):
        """
        Выполняет локальный BA на заданных кейфреймах.

        """
        data = self._collect_ba_data(recent_keyframes, map_points)
        if not data:
            return

        camera_params, points_3d, camera_indices, point_indices, points_2d = data
        ba = BundleAdjustment(self.odometry_calculator.camera_matrix)
        opt_cam_params, opt_points_3d = ba.run_bundle_adjustment(
            camera_params, points_3d, camera_indices, point_indices, points_2d
        )
        self._update_optimized_values(opt_cam_params, opt_points_3d, recent_keyframes, map_points)

    def _collect_ba_data(
        self,
        keyframes: list,
        map_points: list
    ):
        """
        Сбор данных для локального BA (параметры камер, 3D точки, 2D наблюдения).

        """
        camera_params = []
        points_3d = []
        camera_indices = []
        point_indices = []
        points_2d = []

        point_id_to_idx = {}
        idx = 0
        for mp in map_points:
            point_id_to_idx[mp.id] = idx
            points_3d.append(mp.coordinates)
            idx += 1

        for kf in keyframes:
            pose = kf[3]
            R = pose[:, :3]
            t = pose[:, 3]
            rvec, _ = cv2.Rodrigues(R)
            params = np.hstack((rvec.flatten(), t.flatten()))
            camera_params.append(params)

        for cam_idx, kf_data in enumerate(keyframes):
            frame_idx, keypoints, descriptors, pose = kf_data
            for mp in map_points:
                for obs in mp.observations:
                    if obs[0] == frame_idx:
                        kp_idx = obs[1]
                        kp = keypoints[kp_idx]
                        camera_indices.append(cam_idx)
                        point_indices.append(point_id_to_idx[mp.id])
                        points_2d.append(kp.pt)

        if not points_2d:
            self.logger.warning("Нет наблюдений для BA.")
            return None

        return (
            np.array(camera_params),
            np.array(points_3d),
            np.array(camera_indices, dtype=int),
            np.array(point_indices, dtype=int),
            np.array(points_2d)
        )

    def _update_optimized_values(
        self,
        opt_cam_params: np.ndarray,
        opt_points_3d: np.ndarray,
        keyframes: list,
        map_points: list
    ):
        """
        Обновляет позы камер (кейфреймы) и координаты 3D точек после BA.
        """
        for i, kf in enumerate(keyframes):
            frame_idx, kp, desc, _ = kf
            rvec = opt_cam_params[i, :3]
            tvec = opt_cam_params[i, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            pose = np.hstack((R, tvec.reshape(3, 1)))
            keyframes[i] = (frame_idx, kp, desc, pose)

        for i, mp in enumerate(map_points):
            mp.coordinates = opt_points_3d[i]

    def _clean_local_map(
        self,
        map_points: list,
        current_pose: np.ndarray
    ) -> list:
        """
        Вызывает метод из odometry_calculator для очистки карты.

        """
        new_map_points = self.odometry_calculator.clean_local_map(map_points, current_pose)
        self.logger.info(f"Точек на карте после очистки: {len(new_map_points)}")
        return new_map_points

    def process_frame(
        self,
        frame_idx: int,
        current_frame,
        ref_keypoints,
        ref_descriptors: np.ndarray,
        last_pose: np.ndarray,
        map_points: list,
        initialization_completed: bool,
        poses: list,
        keyframes: list
    ):
        """
        Обрабатывает один кадр и запускает основные этапы SLAM/VSLAM.

        """
        curr_keypoints, curr_descriptors = self.feature_extractor.extract_features(current_frame)
        if len(curr_keypoints) == 0:
            self.logger.warning("Не удалось извлечь фичи. Пропускаем кадр.")
            return None

        if not initialization_completed:
            # Если нет опорных ключевых точек – устанавливаем их
            if ref_keypoints is None or ref_descriptors is None:
                self.logger.info("Устанавливаем текущий кадр в качестве опорного для инициализации.")
                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors
                return ref_keypoints, ref_descriptors, last_pose, map_points, initialization_completed

            init_result = self._initialize_map(ref_keypoints, ref_descriptors,
                                               curr_keypoints, curr_descriptors)
            if not init_result:
                return None  # Инициализация не удалась

            R, t, matches, mask_pose = init_result
            initial_pose = np.hstack((R, t))
            poses.append(initial_pose)
            keyframes.append((frame_idx, curr_keypoints, curr_descriptors, initial_pose))
            last_pose = initial_pose
            initialization_completed = True
            self.logger.info("Инициализация прошла успешно.")

            map_points = self._triangulate_initial_points(
                R, t,
                ref_keypoints, curr_keypoints,
                ref_descriptors, curr_descriptors,
                matches, mask_pose,
                frame_idx,
                map_points
            )

            ref_keypoints = curr_keypoints
            ref_descriptors = curr_descriptors

            return ref_keypoints, ref_descriptors, last_pose, map_points, initialization_completed

        else:
            current_pose = self._estimate_pose(map_points, curr_keypoints, curr_descriptors, last_pose)
            if current_pose is None:
                return None  # Позу не удалось восстановить

            poses.append(current_pose)
            last_pose = current_pose

            if self._should_insert_keyframe(keyframes[-1][3], current_pose, frame_idx):
                self._insert_keyframe(
                    frame_idx,
                    curr_keypoints,
                    curr_descriptors,
                    current_pose,
                    keyframes,
                    map_points
                )

                if len(keyframes) >= self.bundle_adjustment_frames:
                    self._run_local_ba(keyframes[-self.bundle_adjustment_frames:], map_points)

            map_points = self._clean_local_map(map_points, current_pose)

            self.odometry_calculator.update_connections_after_pnp(
                map_points, curr_keypoints, curr_descriptors, frame_idx
            )

            ref_keypoints = curr_keypoints
            ref_descriptors = curr_descriptors

            return ref_keypoints, ref_descriptors, last_pose, map_points, initialization_completed