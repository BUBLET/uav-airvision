import logging
import numpy as np
import cv2

from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self,
                 feature_extractor,
                 feature_matcher,
                 odometry_calculator,
                 triangulation_threshold = np.deg2rad(1.0)
                 ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.odometry_calculator = odometry_calculator
        self.triangulation_threshold = triangulation_threshold
        self.logger = logging.getLogger(__name__)
    
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

            if R is None or t is None:
                self.logger.warning("Failed to recover camera pose. Skipping frame.")
                return None

            # Check the median triangulation angle
            median_angle = self.odometry_calculator.check_triangulation_angle(R, t, ref_keypoints, curr_keypoints, matches)
            self.logger.info(f"Median triangulation angle: {np.rad2deg(median_angle):.2f} degrees.")

            if median_angle < self.triangulation_threshold:
                self.logger.warning("Median triangulation angle below threshold. Moving to next frame.")
                return None
            else:
                self.logger.info("Initialization completed.")
                init_completed = True
                # Save the initial pose
                initial_pose = np.hstack((R, t))
                poses.append(initial_pose)
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, initial_pose))
                # Triangulate initial map points
                map_points_array = self.odometry_calculator.triangulate_points(R, t, ref_keypoints, curr_keypoints, matches)
                # Convert to map point structures
                map_points = self.odometry_calculator.convert_points_to_structure(
                    map_points_array, ref_keypoints, matches, ref_descriptors
                )
                # Update reference keypoints and descriptors
                ref_keypoints = curr_keypoints
                ref_descriptors = curr_descriptors
                last_pose = initial_pose
                return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed
        else:
            # Post-initialization: estimate camera pose for subsequent frames
            # Find map points visible in the current frame
            visible_map_points, map_point_indices = self.odometry_calculator.visible_map_points(
                map_points, curr_keypoints, curr_descriptors, last_pose
            )

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

            if not retval or inliers is None or len(inliers) < 4:
                self.logger.warning("Failed to estimate camera pose. Skipping frame")
                return None

            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec

            # Save the current pose
            current_pose = np.hstack((R, t))
            poses.append(current_pose)
            last_pose = current_pose

            # Add new keyframe and update map points periodically or based on criteria
            if frame_idx % 5 == 0:
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, current_pose))
                # Triangulate new map points between the last two keyframes
                new_map_points = self.odometry_calculator.triangulate_new_map_points(
                    keyframes[-2],
                    keyframes[-1],
                    self.feature_matcher,
                    poses
                )
                # Extend the map points list
                map_points.extend(new_map_points)

            # Update reference keypoints and descriptors
            ref_keypoints = curr_keypoints
            ref_descriptors = curr_descriptors

            # Return updated variables
            return ref_keypoints, ref_descriptors, last_pose, map_points, init_completed