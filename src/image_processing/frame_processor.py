import logging
import numpy as np
import cv2
import config
from optimization.ba import BundleAdjustment  # Импортируем BundleAdjustment

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics_logger")

class FrameProcessor:
    def __init__(self,
                 feature_extractor,
                 feature_matcher,
                 odometry_calculator,
                 translation_threshold=config.TRANSLATION_THRESHOLD,
                 rotation_threshold=config.ROTATION_THRESHOLD,
                 triangulation_threshold=config.TRIANGULATION_THRESHOLD,
                 bundle_adjustment_frames=config.BUNDLE_ADJUSTMENT_FRAMES  # Количество кадров для оптимизации
                 ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.odometry_calculator = odometry_calculator
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.triangulation_threshold = triangulation_threshold
        self.bundle_adjustment_frames = bundle_adjustment_frames
        self.logger = logging.getLogger(__name__)
    
    def to_homogeneous(self, pose):
        hom_pose = np.eye(4)
        hom_pose[:3, :3] = pose[:3, :3]
        hom_pose[:3, 3] = pose[:3, 3]
        return hom_pose

    def should_insert_keyframe(self, last_keyframe_pose, current_pose, frame_index, force_keyframe_interval=1):
        if frame_index % force_keyframe_interval == 0:
            self.logger.info(f"Forced keyframe insertion at frame {frame_index}.")
            return True
        
        # Преобразуем позы в однородные координаты
        last_keyframe_pose_hom = self.to_homogeneous(last_keyframe_pose)
        current_pose_hom = self.to_homogeneous(current_pose)

        # Рассчитываем относительное преобразование между позами
        delta_pose = np.linalg.inv(last_keyframe_pose_hom) @ current_pose_hom
        delta_translation = np.linalg.norm(delta_pose[:3, 3])

        # Вычисляем угол вращения через косинус угла
        cos_angle = (np.trace(delta_pose[:3, :3]) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        delta_rotation = np.arccos(cos_angle)

        # Логируем результаты
        if np.isnan(delta_rotation):
            self.logger.warning(f"Delta rotation is NaN. Cosine value: {cos_angle}")
            return False
        else:
            self.logger.info(f"Frame: {frame_index}, Delta translation: {delta_translation}, "
                            f"Delta rotation (deg): {np.rad2deg(delta_rotation)}")

        # Проверяем превышение порогов
        if delta_translation > self.translation_threshold or delta_rotation > self.rotation_threshold:
            return True
        else:
            return False

    def update_optimized_values(self, optimized_camera_params, optimized_points_3d, keyframes, map_points):
            """
            Обновляет параметры камер и 3D точек после оптимизации.

            Параметры:
            - optimized_camera_params: оптимизированные параметры камер.
            - optimized_points_3d: оптимизированные 3D точки.
            - keyframes: список кейфреймов для обновления.
            - map_points: список точек карты для обновления.
            """
            # Обновление поз камер
            for idx, kf in enumerate(keyframes):
                rvec = optimized_camera_params[idx, :3]
                tvec = optimized_camera_params[idx, 3:6]
                R, _ = cv2.Rodrigues(rvec)
                pose = np.hstack((R, tvec.reshape(3, 1)))
                frame_idx, keypoints, descriptors, _ = kf
                keyframes[idx] = (frame_idx, keypoints, descriptors, pose)

            # Обновление координат точек карты
            for mp_idx, mp in enumerate(map_points):
                mp.coordinates = optimized_points_3d[mp_idx]

    def collect_bundle_adjustment_data(self, keyframes, map_points):
        """
        Собирает данные для оптимизации Bundle Adjustment.

        Параметры:
        - keyframes: список последних N кейфреймов.
        - map_points: текущие точки карты.

        Возвращает:
        - camera_params: параметры камер.
        - points_3d: координаты 3D точек.
        - camera_indices: индексы камер для наблюдений.
        - point_indices: индексы точек для наблюдений.
        - points_2d: наблюдаемые 2D точки.
        """
        import numpy as np

        camera_params = []
        points_3d = []
        camera_indices = []
        point_indices = []
        points_2d = []

        # Создаем отображение от ID точки карты к индексу
        point_id_to_idx = {}
        idx = 0
        for mp in map_points:
            point_id_to_idx[mp.id] = idx
            points_3d.append(mp.coordinates)
            idx += 1

        # Сбор параметров камер из кейфреймов
        for kf in keyframes:
            pose = kf[3]
            R = pose[:, :3]
            t = pose[:, 3]
            rvec, _ = cv2.Rodrigues(R)
            camera_param = np.hstack((rvec.flatten(), t.flatten()))
            camera_params.append(camera_param)

        # Сбор наблюдений
        for cam_idx, kf in enumerate(keyframes):
            frame_idx, keypoints, descriptors, pose = kf
            for mp in map_points:
                for obs in mp.observations:
                    if obs[0] == frame_idx:
                        kp_idx = obs[1]
                        kp = keypoints[kp_idx]
                        points_2d.append(kp.pt)
                        camera_indices.append(cam_idx)
                        point_indices.append(point_id_to_idx[mp.id])

        if not points_2d:
            self.logger.warning("No observations found for bundle adjustment.")
            return None

        camera_params = np.array(camera_params)
        points_3d = np.array(points_3d)
        camera_indices = np.array(camera_indices, dtype=int)
        point_indices = np.array(point_indices, dtype=int)
        points_2d = np.array(points_2d)

        return camera_params, points_3d, camera_indices, point_indices, points_2d

    def process_frame(self,
                      frame_idx,
                      current_frame,
                      ref_keypoints,
                      ref_descriptors,
                      last_pose,
                      map_points,
                      initialization_completed,
                      poses,
                      keyframes
                      ):
        init_completed = initialization_completed

        # Извлечение ключевых точек и дескрипторов из текущего кадра
        curr_keypoints, curr_descriptors = self.feature_extractor.extract_features(current_frame)
        if len(curr_keypoints) == 0:
            self.logger.warning("Failed to detect keypoints in the current frame. Skipping frame.")
            return None
        

        if not init_completed:
            if ref_keypoints is None or ref_descriptors is None:
                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors
                self.logger.info("Reference keyframe set.")
                return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed
            
            # Сопоставление особенностей между опорным и текущим кадром
            matches = self.feature_matcher.match_features(ref_descriptors, curr_descriptors)
            if len(matches) < 4:
                self.logger.warning(f"Not enough matches ({len(matches)}) for odometry computation. Skipping frame.")
                return None

            # Вычисление матриц Essential и Homography
            E_result = self.odometry_calculator.calculate_essential_matrix(ref_keypoints, curr_keypoints, matches)
            H_result = self.odometry_calculator.calculate_homography_matrix(ref_keypoints, curr_keypoints, matches)

            if E_result is None or H_result is None:
                self.logger.warning("Failed to compute E or H matrices. Skipping frame.")
                return None

            E, mask_E, error_E = E_result
            H, mask_H, error_H = H_result

            # Выбор между E и H на основе ошибки
            total_error = error_E + error_H
            if total_error == 0:
                self.logger.warning("Total error is zero. Skipping frame.")
                return None

            H_ratio = error_H / total_error
            use_homography = H_ratio > config.HOMOGRAPHY_INLIER_RATIO

            if use_homography:
                self.logger.info("Homography matrix chosen for decomposition.")
                R, t, mask_pose = self.odometry_calculator.decompose_homography(H, ref_keypoints, curr_keypoints, matches)
            else:
                self.logger.info("Essential matrix chosen for decomposition.")
                R, t, mask_pose = self.odometry_calculator.decompose_essential(E, ref_keypoints, curr_keypoints, matches)

            if R is None or t is None:
                self.logger.warning("Failed to recover camera pose. Skipping frame.")
                return None

            # Проверка угла триангуляции
            median_angle = self.odometry_calculator.check_triangulation_angle(R, t, ref_keypoints, curr_keypoints, matches)
            self.logger.info(f"Median triangulation angle: {np.rad2deg(median_angle):.2f} degrees.")

            if median_angle < self.triangulation_threshold:
                self.logger.warning("Median triangulation angle below threshold. Moving to next frame.")
                return None
            else:
                self.logger.info("Initialization completed.")
                init_completed = True
                initial_pose = np.hstack((R, t))
                poses.append(initial_pose)
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, initial_pose))
                last_pose = initial_pose

                # Триангуляция начальных точек карты
                map_points_array, inlier_matches = self.odometry_calculator.triangulate_points(
                    R, t, ref_keypoints, curr_keypoints, matches, mask_pose
                )

                # Преобразование в структуру map points
                ref_frame_idx = frame_idx - 1
                curr_frame_idx = frame_idx

                map_points = self.odometry_calculator.convert_points_to_structure(
                    map_points_array,
                    ref_keypoints,
                    curr_keypoints,
                    inlier_matches,
                    ref_descriptors,
                    curr_descriptors,
                    ref_frame_idx,
                    curr_frame_idx
                )

                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors

                return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed
        else:
            # Оценка позы камеры для последующих кадров
            visible_map_points, map_point_indices = self.odometry_calculator.visible_map_points(
                map_points, curr_keypoints, curr_descriptors, last_pose
            )

            self.logger.info(f"Number of visible map points: {len(visible_map_points)}")

            if len(visible_map_points) < 4:
                self.logger.warning("Not enough visible map points for pose estimation. Skipping frame.")
                return None

            # Подготовка соответствий 2D-3D
            object_points = np.array([mp.coordinates for mp in visible_map_points], dtype=np.float32)
            image_points = np.array([curr_keypoints[idx].pt for idx in map_point_indices], dtype=np.float32)

            # Оценка позы камеры с помощью PnP
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.odometry_calculator.camera_matrix,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            self.logger.info(f"Number of inliers in PnP: {len(inliers) if inliers is not None else 0}")

            if not retval or inliers is None or len(inliers) < 4:
                self.logger.warning("Failed to estimate camera pose. Skipping frame")
                return None
            
            R, _ = cv2.Rodrigues(rvec)
            t = tvec

            current_pose = np.hstack((R, t))
            self.logger.info(f"Current pose at frame {frame_idx}:\n{current_pose}")

            if np.allclose(current_pose, last_pose):
                self.logger.warning("Current pose did not change from last pose.")
            else:
                self.logger.info("Current pose updated.")

            poses.append(current_pose)
            last_pose = current_pose

            # Добавление нового кейфрейма и обновление map points
            if self.should_insert_keyframe(last_pose, current_pose, frame_idx):
                self.logger.info(f"Inserting new keyframe at frame {frame_idx}")
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, current_pose))
                
                # Сопоставление с предыдущим ключевым кадром
                prev_keyframe = keyframes[-2]
                matches = self.feature_matcher.match_features(prev_keyframe[2], curr_descriptors)
                inlier_matches = self.odometry_calculator.get_inliers_epipolar(
                    prev_keyframe[1], curr_keypoints, matches)
                self.logger.info(f"Found {len(inlier_matches)} inlier matches for triangulation.")
                
                if len(inlier_matches) < 8:
                    self.logger.warning(f"Not enough matches ({len(inlier_matches)}) for triangulation. Skipping keyframe insertion.")
                else:
                    # Триангуляция новых точек карты
                    new_map_points = self.odometry_calculator.triangulate_new_map_points(
                        prev_keyframe,
                        keyframes[-1],
                        inlier_matches,
                    )
                    map_points.extend(new_map_points)
                    self.logger.info(f"Total map points: {len(map_points)}")

                # Проверка наличия достаточного количества кейфреймов для BA
                if len(keyframes) >= self.bundle_adjustment_frames:
                    # Сбор данных для BA
                    ba_data = self.collect_bundle_adjustment_data(keyframes[-self.bundle_adjustment_frames:], map_points)
                    if ba_data:
                        camera_params, points_3d, camera_indices, point_indices, points_2d = ba_data

                        # Создание экземпляра BA
                        ba = BundleAdjustment(self.odometry_calculator.camera_matrix)

                        # Запуск BA
                        optimized_camera_params, optimized_points_3d = ba.run_bundle_adjustment(
                            camera_params, points_3d, camera_indices, point_indices, points_2d
                        )

                        # Обновление поз и точек карты
                        self.update_optimized_values(optimized_camera_params, optimized_points_3d, keyframes[-self.bundle_adjustment_frames:], map_points)
            
            map_points = self.odometry_calculator.clean_local_map(map_points, current_pose)
            self.logger.info(f"Number of map points after cleaning: {len(map_points)}")

            self.odometry_calculator.update_connections_after_pnp(
                map_points, curr_keypoints, curr_descriptors, frame_idx
            )

            ref_keypoints = curr_keypoints
            ref_descriptors = curr_descriptors

            return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed

  