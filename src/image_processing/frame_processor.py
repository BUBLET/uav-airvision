import logging
import numpy as np
import cv2

from image_processing import FeatureExtractor, FeatureMatcher, OdometryCalculator

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self,
                 feature_extractor,
                 feature_matcher,
                 odometry_calculator,
                 translation_threshold=0.01,
                 rotation_threshold=np.deg2rad(10),
                 triangulation_threshold = np.deg2rad(0.5)
                 ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.odometry_calculator = odometry_calculator
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.triangulation_threshold = triangulation_threshold
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

        # Extract keypoints and descriptors from the current frame
        curr_keypoints, curr_descriptors = self.feature_extractor.extract_features(current_frame)
        if len(curr_keypoints) == 0:
            self.logger.warning("Failed to detect keypoints in the current frame. Skipping frame.")
            return None
        
        # Отображаем ключевые точки на текущем кадре
        img_with_keypoints = cv2.drawKeypoints(current_frame, curr_keypoints, None, color=(0, 255, 0))
        cv2.imshow('Keypoints', img_with_keypoints)
        cv2.waitKey(1)  # Задержка в 1 миллисекунду, чтобы окно обновлялось
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

        if not init_completed:
            # Match features between the reference frame and the current frame
            matches = self.feature_matcher.match_features(ref_descriptors, curr_descriptors)
            if len(matches) < 8:
                self.logger.warning(f"Not enough matches ({len(matches)}) for odometry computation. Skipping frame.")
                return None

            # Compute Essential and Homography matrices
            E_result = self.odometry_calculator.calculate_essential_matrix(ref_keypoints, curr_keypoints, matches)
            H_result = self.odometry_calculator.calculate_homography_matrix(ref_keypoints, curr_keypoints, matches)

            if E_result is None or H_result is None:
                self.logger.warning("Failed to compute E or H matrices. Skipping frame.")
                return None

            E, mask_E, error_E = E_result
            H, mask_H, error_H = H_result

            # Decide between E and H based on symmetric transfer error
            total_error = error_E + error_H
            if total_error == 0:
                self.logger.warning("Total error is zero. Skipping frame.")
                return None

            H_ratio = error_H / total_error
            use_homography = H_ratio > 0.45

            if use_homography:
                self.logger.info("Homography matrix chosen for decomposition.")
                R, t, mask_pose = self.odometry_calculator.decompose_homography(H, ref_keypoints, curr_keypoints, matches)
            else:
                self.logger.info("Essential matrix chosen for decomposition.")
                R, t, mask_pose = self.odometry_calculator.decompose_essential(E, ref_keypoints, curr_keypoints, matches)

            # После разложения E или H и получения mask_pose
            if R is None or t is None:
                self.logger.warning("Failed to recover camera pose. Skipping frame.")
                return None

            # Проверяем средний угол триангуляции
            median_angle = self.odometry_calculator.check_triangulation_angle(R, t, ref_keypoints, curr_keypoints, matches)
            self.logger.info(f"Median triangulation angle: {np.rad2deg(median_angle):.2f} degrees.")

            if median_angle < self.triangulation_threshold:
                self.logger.warning("Median triangulation angle below threshold. Moving to next frame.")
                return None
            else:
                self.logger.info("Initialization completed.")
                init_completed = True
                # Сохраняем начальную позу
                initial_pose = np.hstack((R, t))
                poses.append(initial_pose)
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, initial_pose))
                last_pose = initial_pose

                # Триангулируем начальные точки карты
                map_points_array, inlier_matches = self.odometry_calculator.triangulate_points(
                    R, t, ref_keypoints, curr_keypoints, matches, mask_pose
                )

                # Преобразуем в структуру map points
                ref_frame_idx = frame_idx - 1  # Индекс опорного кадра
                curr_frame_idx = frame_idx     # Индекс текущего кадра

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

                # Обновляем опорные ключевые точки и дескрипторы
                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors

                return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed
        else:
            # Post-initialization: estimate camera pose for subsequent frames
            # Find map points visible in the current frame
            visible_map_points, map_point_indices = self.odometry_calculator.visible_map_points(
                map_points, curr_keypoints, curr_descriptors, last_pose
            )

            self.logger.info(f"Number of visible map points: {len(visible_map_points)}")

            if len(visible_map_points) < 4:
                self.logger.warning("Not enough visible map points for pose estimation. Skipping frame.")
                return None

            # Prepare 2D-3D correspondences
            object_points = np.array([mp.coordinates for mp in visible_map_points], dtype=np.float32)  # 3D points
            image_points = np.array([curr_keypoints[idx].pt for idx in map_point_indices], dtype=np.float32)  # 2D points

            # Estimate camera pose using PnP with RANSAC
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
            

            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec

            # Save the current pose
            current_pose = np.hstack((R, t))
            self.logger.info(f"Current pose at frame {frame_idx}:\n{current_pose}")

            if np.allclose(current_pose, last_pose):
                self.logger.warning("Current pose did not change from last pose.")
            else:
                self.logger.info("Current pose updated.")

            poses.append(current_pose)
            last_pose = current_pose

            
            # Add new keyframe and update map points periodically or based on criteria
            if self.should_insert_keyframe(last_pose, current_pose, frame_idx):
                self.logger.info(f"Inserting new keyframe at frame {frame_idx}")
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, current_pose))
                
                # Сопоставление с предыдущим ключевым кадром
                prev_keyframe = keyframes[-2]
                matches = self.feature_matcher.match_features(prev_keyframe[2], curr_descriptors)
                inlier_matches = self.odometry_calculator.get_inliers_epipolar(
                    prev_keyframe[1], curr_keypoints, matches)
                self.logger.info(f"Найдено {len(inlier_matches)} inlier соответствий для триангуляции.")
                
                # Проверяем достаточность соответствий
                if len(inlier_matches) < 8:
                    self.logger.warning(f"Not enough matches ({len(inlier_matches)}) for triangulation. Skipping keyframe insertion.")
                else:
                    # Triangulate new map points between the last two keyframes using inlier_matches
                    new_map_points = self.odometry_calculator.triangulate_new_map_points(
                        prev_keyframe,
                        keyframes[-1],
                        inlier_matches,
                        map_points
                    )
                    # Extend the map points list
                    map_points.extend(new_map_points)
                    self.logger.info(f"Общее количество точек карты: {len(map_points)}")

            map_points = self.odometry_calculator.clean_local_map(map_points, current_pose)
            self.logger.info(f"Number of map points after cleaning: {len(map_points)}")

            self.odometry_calculator.update_connections_after_pnp(
                map_points, curr_keypoints, curr_descriptors, frame_idx
            )

            # Update reference keypoints and descriptors
            ref_keypoints = curr_keypoints
            ref_descriptors = curr_descriptors

            # Return updated variables
            return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed