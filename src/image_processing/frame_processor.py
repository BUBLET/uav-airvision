import logging
import numpy as np
import cv2
import config

from optimization.ba import BundleAdjustment
from image_processing.lk_tracker import LKTracker

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics_logger")


def remove_points_with_large_reprojection_error(map_points, keyframes, camera_matrix, max_reproj_error=5.0):
    """Удаляет точки, у которых средняя ошибка репроекции по всем наблюдениям > max_reproj_error."""
    good_points = []
    for mp in map_points:
        total_error = 0.0
        valid_count = 0

        for (frame_idx, kp_idx) in mp.observations:
            # Ищем соответствующий кортеж в keyframes: (frame_idx, keypoints, descriptors, pose)
            kf = next((k for k in keyframes if k[0] == frame_idx), None)
            if kf is None:
                continue

            pose = kf[3]  # 3x4
            keypoints = kf[1]
            if kp_idx >= len(keypoints):
                continue

            kp = keypoints[kp_idx]
            pt3d = mp.coordinates.reshape(3, 1)

            proj_matrix = camera_matrix @ pose
            pt2d_hom = proj_matrix @ np.vstack((pt3d, [[1.0]]))
            if abs(pt2d_hom[2, 0]) < 1e-9:
                continue

            pt2d = (pt2d_hom[:2] / pt2d_hom[2]).ravel()
            keypt = np.array(kp.pt)
            error = np.linalg.norm(pt2d - keypt)
            total_error += error
            valid_count += 1

        if valid_count > 0:
            mean_error = total_error / valid_count
            if mean_error < max_reproj_error:
                good_points.append(mp)
        else:
            # Если точка нигде не наблюдалась нормально — можно исключить
            pass

    return good_points


def remove_points_with_few_observations(map_points, min_obs=2):
    """Удаляет точки, у которых слишком мало наблюдений (observations)."""
    return [mp for mp in map_points if len(mp.observations) >= min_obs]


def limit_map_points(map_points, max_points=1000):
    """
    Если общее число точек слишком велико, оставляет только max_points
    самых «качественных» (здесь сортировка по числу observations).
    """
    if len(map_points) <= max_points:
        return map_points
    map_points_sorted = sorted(map_points,
                               key=lambda mp: len(mp.observations),
                               reverse=True)
    return map_points_sorted[:max_points]



class FrameProcessor:
    def __init__(self,
                 feature_extractor,
                 feature_matcher,
                 odometry_calculator,
                 ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.odometry_calculator = odometry_calculator

        self.translation_threshold = config.TRANSLATION_THRESHOLD
        self.rotation_threshold = config.ROTATION_THRESHOLD
        self.triangulation_threshold = config.TRIANGULATION_THRESHOLD
        self.bundle_adjustment_frames = config.BUNDLE_ADJUSTMENT_FRAMES
        self.logger = logging.getLogger(__name__)

        # KLT-трекер
        self.lk_tracker = LKTracker()

        self.orb_interval = config.ORB_INTERVAL  
        self.force_keyframe_interval = config.FORCE_KEYFRAME_INTERVAL 
        self.keyframe_BA_interval = config.KEYFRAME_BA_INTERVAL 

        # Храним предыдущий кадр/точки для LK
        self.prev_frame = None
        self.prev_points = None

        # Настройки фильтрации
        self.max_reproj_error = config.MAX_REPROJ_ERROR
        self.min_observations = config.MIN_OBSERVATIONS
        self.max_points = config.MAX_POINTS

    def to_homogeneous(self, pose):
        hom_pose = np.eye(4)
        hom_pose[:3, :3] = pose[:3, :3]
        hom_pose[:3, 3] = pose[:3, 3]
        return hom_pose

    def should_insert_keyframe(self, last_keyframe_pose, current_pose, frame_index):
        """
        Логика вставки нового кейфрейма:
        - каждые force_keyframe_interval кадров (принудительно),
        - либо если превышаются пороги смещения/вращения.
        """
        if frame_index % config.FORCE_KEYFRAME_INTERVAL == 0:
            self.logger.info(f"Forced keyframe insertion at frame {frame_index}.")
            return True
        
        # Преобразуем позы в однородные координаты
        last_keyframe_pose_hom = self.to_homogeneous(last_keyframe_pose)
        current_pose_hom = self.to_homogeneous(current_pose)

        # Рассчитываем относительное преобразование
        delta_pose = np.linalg.inv(last_keyframe_pose_hom) @ current_pose_hom
        delta_translation = np.linalg.norm(delta_pose[:3, 3])

        # Угол вращения
        cos_angle = (np.trace(delta_pose[:3, :3]) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        delta_rotation = np.arccos(cos_angle)

        if np.isnan(delta_rotation):
            self.logger.warning(f"Delta rotation is NaN. Cosine value: {cos_angle}")
            return False
        else:
            self.logger.info(f"Frame: {frame_index}, Delta translation: {delta_translation}, "
                             f"Delta rotation (deg): {np.rad2deg(delta_rotation)}")

        # Проверяем превышение порогов
        if (delta_translation > self.translation_threshold) or (delta_rotation > self.rotation_threshold):
            return True
        else:
            return False

    def update_optimized_values(self, optimized_camera_params, optimized_points_3d, keyframes, map_points):
        
        """
        Обновляет параметры камер и 3D точек после Bundle Adjustment.
        """
        for idx, kf in enumerate(keyframes):
            rvec = optimized_camera_params[idx, :3]
            tvec = optimized_camera_params[idx, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            pose = np.hstack((R, tvec.reshape(3, 1)))
            frame_idx, keypoints, descriptors, _ = kf
            keyframes[idx] = (frame_idx, keypoints, descriptors, pose)

        for mp_idx, mp in enumerate(map_points):
            mp.coordinates = optimized_points_3d[mp_idx]

    def collect_bundle_adjustment_data(self, keyframes, map_points):
        """
        Собирает данные (camera_params, points_3d...) для BA.
        """
        camera_params = []
        points_3d = []
        camera_indices = []
        point_indices = []
        points_2d = []

        # Индексация map_points
        point_id_to_idx = {}
        idx = 0
        for mp in map_points:
            point_id_to_idx[mp.id] = idx
            points_3d.append(mp.coordinates)
            idx += 1

        # Сбор параметров камер (rvec, tvec)
        for kf in keyframes:
            pose = kf[3]
            R = pose[:, :3]
            t = pose[:, 3]
            rvec, _ = cv2.Rodrigues(R)
            camera_param = np.hstack((rvec.flatten(), t.flatten()))
            camera_params.append(camera_param)

        # Сбор 2D-наблюдений
        for cam_idx, kf in enumerate(keyframes):
            frame_idx, keypoints, descriptors, pose = kf
            for mp in map_points:
                for obs in mp.observations:
                    if obs[0] == frame_idx:
                        kp_idx = obs[1]
                        if kp_idx < len(keypoints):
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

    def filter_map_points(self, map_points, keyframes):
        
        """
        Применяем несколько фильтров подряд:
        - убираем точки с большой ошибкой репроекции,
        - убираем с малым числом наблюдений,
        - ограничиваем общее число (макс. self.max_points).
        """
        map_points = remove_points_with_large_reprojection_error(
            map_points, keyframes, 
            self.odometry_calculator.camera_matrix,
            max_reproj_error=self.max_reproj_error
        )

        map_points = remove_points_with_few_observations(map_points, self.min_observations)
        map_points = limit_map_points(map_points, self.max_points)
        return map_points

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
        """
        Обрабатывает очередной кадр.
        Возвращает (ref_keypoints, ref_descriptors, last_pose, map_points, init_completed).
        """

        init_completed = initialization_completed
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        use_orb_now = (frame_idx % self.orb_interval == 0)

        if not init_completed:
            use_orb_now = True

        if not init_completed:
            # Извлекаем ключевые точки и дескрипторы
            curr_keypoints, curr_descriptors = self.feature_extractor.extract_features(current_frame)
            if len(curr_keypoints) == 0:
                self.logger.warning("Failed to detect keypoints in the current frame. Skipping frame.")
                return None

            # Отображение ключевых точек (как и раньше)
            # img_with_keypoints = cv2.drawKeypoints(current_frame, curr_keypoints, None, color=(0, 255, 0))
            # cv2.imshow('Keypoints', img_with_keypoints)
            # cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     exit()

            if ref_keypoints is None or ref_descriptors is None:
                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors
                self.logger.info("Reference keyframe set.")
                # Сохраняем кадр/точки для LK в будущем (после init)
                self.prev_frame = current_gray
                self.prev_points = np.array([kp.pt for kp in curr_keypoints], dtype=np.float32)
                return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed

            # Сопоставляем с опорным
            matches = self.feature_matcher.match_features(ref_descriptors, curr_descriptors)
            if len(matches) < 4:
                self.logger.warning(f"Not enough matches ({len(matches)}) for odometry computation. Skipping frame.")
                return None

            # Вычисляем Essential/Homography
            E_result = self.odometry_calculator.calculate_essential_matrix(ref_keypoints, curr_keypoints, matches)
            H_result = self.odometry_calculator.calculate_homography_matrix(ref_keypoints, curr_keypoints, matches)

            if E_result is None or H_result is None:
                self.logger.warning("Failed to compute E or H matrices. Skipping frame.")
                return None

            E, mask_E, error_E = E_result
            H, mask_H, error_H = H_result

            total_error = error_E + error_H
            if total_error == 0:
                self.logger.warning("Total error is zero. Skipping frame.")
                return None

            H_ratio = error_H / total_error
            use_homography = (H_ratio > config.HOMOGRAPHY_INLIER_RATIO)

            # Декомпозиция
            if use_homography:
                self.logger.info("Homography matrix chosen for decomposition.")
                metrics_logger.info(f"[MATRIX] Homography matrix chosen for decomposition.")
                R, t, mask_pose = self.odometry_calculator.decompose_homography(
                    H, ref_keypoints, curr_keypoints, matches
                )
            else:
                self.logger.info("Essential matrix chosen for decomposition.")
                metrics_logger.info(f"[MATRIX] Essential matrix chosen for decomposition.")
                R, t, mask_pose = self.odometry_calculator.decompose_essential(
                    E, ref_keypoints, curr_keypoints, matches
                )

            if R is None or t is None:
                self.logger.warning("Failed to recover camera pose. Skipping frame.")
                return None
            # t = t * config.INIT_SCALE
            # self.logger.info(f"Applied scale factor {config.INIT_SCALE:.3f} to translation. "
            #                 f"New t={t.ravel()}")
            # Проверка угла триангуляции
            median_angle = self.odometry_calculator.check_triangulation_angle(
                R, t, ref_keypoints, curr_keypoints, matches
            )
            self.logger.info(f"Median triangulation angle: {np.rad2deg(median_angle):.2f} degrees.")
            metrics_logger.info(f"[INIT] Frame {frame_idx}, median_angle_deg={np.rad2deg(median_angle):.2f}, "
                    f"num_inliers={np.count_nonzero(mask_pose)}")
            
            if median_angle < self.triangulation_threshold:
                self.logger.warning("Median triangulation angle below threshold. Moving to next frame.")
                return None
            else:
                # Инициализация завершена
                self.logger.info("Initialization completed.")
                init_completed = True

                initial_pose = np.hstack((R, t))
                poses.append(initial_pose)
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, initial_pose))
                last_pose = initial_pose

                # Триангулируем начальные точки
                map_points_array, inlier_matches = self.odometry_calculator.triangulate_points(
                    R, t, ref_keypoints, curr_keypoints, matches, mask_pose
                )

                pts3d = map_points_array

                if pts3d.size > 0:
                    mean_depth = np.mean(pts3d[:, 2])
                    assumed_mean_depth = config.ASSUMED_MEAN_DEPTH_DURING_INIT
                    scale = assumed_mean_depth / mean_depth
                    self.logger.info(f"Calculated scale factor: {scale:.3f}")

                    t = t * scale
                    map_points_array = pts3d * scale

                    initial_pose = np.hstack((R, t))
                    poses[-1] = initial_pose
                    last_pose = initial_pose
                else:
                    self.logger.warning("No 3d points triangulated")
                
                # Преобразуем в структуру
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

                # Обновляем ref
                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors

                # Сохраняем кадр/точки для LK
                self.prev_frame = current_gray
                self.prev_points = np.array([kp.pt for kp in curr_keypoints], dtype=np.float32)

                return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed

        if use_orb_now:
            # Делаем ORB-детектирование
            curr_keypoints, curr_descriptors = self.feature_extractor.extract_features(current_frame)
            if len(curr_keypoints) == 0:
                self.logger.warning("No keypoints found. Skipping frame.")
                return None

            img_with_keypoints = cv2.drawKeypoints(current_frame, curr_keypoints, None, color=(0, 255, 0))
            cv2.imshow('Keypoints', img_with_keypoints)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()

            # 2D-3D (PnP)
            visible_map_points, map_point_indices = self.odometry_calculator.visible_map_points(
                map_points, curr_keypoints, curr_descriptors, last_pose
            )
            self.logger.info(f"Number of visible map points: {len(visible_map_points)}")

            if len(visible_map_points) < 4:
                self.logger.warning("Not enough visible map points for pose estimation. Skipping frame.")
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
            self.logger.info(f"Number of inliers in PnP: {len(inliers) if inliers is not None else 0}")
            metrics_logger.info(f"[PnP] Frame {frame_idx}, inliers_pnp={len(inliers) if inliers is not None else 0}")

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

            # Сохраняем кадр/точки для LK
            self.prev_frame = current_gray
            self.prev_points = np.array([kp.pt for kp in curr_keypoints], dtype=np.float32)

        else:
            curr_keypoints = []
            curr_descriptors = None

            if self.prev_frame is not None and self.prev_points is not None and len(self.prev_points) > 0:
                # Запуск трекера
                prev_pts_good, curr_pts_good = self.lk_tracker.track(
                    self.prev_frame, current_gray, self.prev_points
                )
                self.logger.info(f"LK tracking: from {len(self.prev_points)} -> {len(curr_pts_good)} good points")

                # Можно визуализировать:
                # disp = current_frame.copy()
                # for (x, y) in curr_pts_good:
                #     cv2.circle(disp, (int(x), int(y)), 3, (0, 255, 0), -1)
                # cv2.imshow('LK tracking', disp)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                #     exit()

                # Здесь упрощённо: мы НЕ делаем PnP на этих трекнутых точках
                # Сохраняем result для следующего кадра
                self.prev_frame = current_gray
                self.prev_points = curr_pts_good
            else:
                # Нет точек для LK
                self.logger.warning("No points to track with LK.")
                self.prev_frame = current_gray
                self.prev_points = None
            current_pose = last_pose

        # Добавление нового кейфрейма (только если этот кадр был ORB/PNP)
        if use_orb_now:
            if self.should_insert_keyframe(last_pose, current_pose, frame_idx):
                self.logger.info(f"Inserting new keyframe at frame {frame_idx}")
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, current_pose))

                # Сопоставление с предыдущим ключевым кадром
                if len(keyframes) > 1:
                    prev_keyframe = keyframes[-2]
                    matches = self.feature_matcher.match_features(prev_keyframe[2], curr_descriptors)
                    inlier_matches = self.odometry_calculator.get_inliers_epipolar(
                        prev_keyframe[1], curr_keypoints, matches
                    )
                    self.logger.info(f"Found {len(inlier_matches)} inlier matches for triangulation.")

                    if len(inlier_matches) >= 8:
                        # Триангуляция новых точек карты
                        new_map_points = self.odometry_calculator.triangulate_new_map_points(
                            prev_keyframe,
                            keyframes[-1],
                            inlier_matches
                        )
                        map_points.extend(new_map_points)
                        self.logger.info(f"Total map points: {len(map_points)}")
                    else:
                        self.logger.warning(f"Not enough matches ({len(inlier_matches)}) for triangulation.")

    
                if (len(keyframes) >= self.bundle_adjustment_frames
                        and (len(keyframes) % self.keyframe_BA_interval == 0)):
                    ba_data = self.collect_bundle_adjustment_data(keyframes[-self.bundle_adjustment_frames:], map_points)
                    if ba_data:
                        camera_params, points_3d, camera_indices, point_indices, points_2d = ba_data

                        # Запуск BA
                        ba = BundleAdjustment(self.odometry_calculator.camera_matrix)
                        optimized_camera_params, optimized_points_3d = ba.run_bundle_adjustment(
                            camera_params, points_3d, camera_indices, point_indices, points_2d
                        )
                        # Обновляем позы и точки
                        self.update_optimized_values(
                            optimized_camera_params, optimized_points_3d,
                            keyframes[-self.bundle_adjustment_frames:], map_points
                        )

        # Очищаем лишние/плохие точки
        map_points = self.odometry_calculator.clean_local_map(map_points, current_pose)
        map_points = self.filter_map_points(map_points, keyframes)
        self.logger.info(f"Number of map points after cleaning: {len(map_points)}")

        # Вызываем update_connections только если есть дескрипторы
        if curr_descriptors is not None and isinstance(curr_descriptors, np.ndarray) and curr_descriptors.size > 0:
            self.odometry_calculator.update_connections_after_pnp(
                map_points, curr_keypoints, curr_descriptors, frame_idx
            )
        # Возвращаем обновлённые данные
        return curr_keypoints, curr_descriptors, last_pose, map_points, init_completed
