import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator
from error_correction.error_correction import ErrorCorrector

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Задаём путь к видеофайлу
    video_path = "datasets/video1.mp4"

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("download video error")
        return

    # Читаем первый кадр и устанавливаем его как опорный
    ret, reference_frame = cap.read()
    if not ret:
        logger.error("no first frame")
        return

    # Получаем размеры изображения
    frame_height, frame_width = reference_frame.shape[:2]

    # Инициализируем объекты для обработки
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    odometry_calculator = OdometryCalculator(image_width=frame_width, image_height=frame_height)

    frame_idx = 1 # Индекс текущего кадра
    initialization_completed = False # Флаг инициализации
    triangulation_threshold = np.deg2rad(1.0) # Порог угла триангуляции

    ref_keypoints, ref_descriptors = feature_extractor.extract_features(reference_frame)
    if len(ref_keypoints) == 0:
        logger.error("no keypoints")
        return
    
    # Хранение мап поинтс, кейфрамес и poses
    map_points = None
    keyframes = []
    poses = []

    while True:
        # Читаем следующий кадр из видео
        ret, current_frame = cap.read()
        if not ret:
            logger.info("complete")
            break
        
        frame_idx += 1

        # Извлекаем ключевые точки и дескрипторы для текущего кадра
        curr_keypoints, curr_descriptors = feature_extractor.extract_features(current_frame)
        if len(curr_keypoints) == 0:
            logger.warning("no keypoints for this frame")
            continue

        if not initialization_completed:
            # Сопоставляем ключевые точки между предыдущим и текущим кадром
            matches = feature_matcher.match_features(ref_descriptors, curr_descriptors)
            if len(matches) < 8:
                logger.warning(f"matches not enough ({len(matches)}) for calculate pos")
                continue

            # Вычисляем матрицы Essential и Homography
            E_result = odometry_calculator.calculate_essential_matrix(ref_keypoints, curr_keypoints, matches)
            H_result = odometry_calculator.calculate_homography_matrix(ref_keypoints, curr_keypoints, matches)
            
            if E_result is None or H_result is None:
                logger.warning("cant calculate E and H matrix")
                continue
            E, mask_E, error_E = E_result
            H, mask_H, error_H = H_result

            # Вычисляем отношение ошибок для выбора матрицы
            total_error = error_E + error_H
            if total_error == 0:
                logger.warning("sum of error 0")
                continue

            H_ratio = error_H / total_error
            use_homography = H_ratio > 0.45

            if use_homography:
                logger.info("H matrix chosen")
                R, t, mask_pose = odometry_calculator.decompose_homography(H, ref_keypoints, curr_keypoints, matches)
            else:
                logger.info("E matix chosen")
                R, t, mask_pose = odometry_calculator.decompose_essential(E, ref_keypoints, curr_keypoints, matches)
            
            if R is None or t is None:
                logger.warning("cant calculate pose for frame")


            # Проверяем угол
            median_angle = odometry_calculator.check_triangulation_angle(R, t, ref_keypoints, curr_keypoints, matches)

            if median_angle < triangulation_threshold:
                logger.warning("median angle < threshold")
                continue
        else:
            # После инициализации продолжаем оценивать позу камеры
            # Используем последнюю позу камеры
            last_pose = poses[-1]

            # Находим map points, которые находятся в поле зрения
            visible_map_points, map_point_indices = odometry_calculator.find_visible_map_points(
                map_points, curr_keypoints, last_pose, curr_descriptors
            )

            if len(visible_map_points) < 3:
                logger.warning("Недостаточно видимых map points для оценки позы. Пропуск кадра.")
                continue

            # Создаем списки 2D-3D соответствий
            object_points = np.array([mp.coordinates for mp in visible_map_points], dtype=np.float32)  # 3D точки
            image_points = np.array([curr_keypoints[idx].pt for idx in map_point_indices], dtype=np.float32)  # 2D точки

            # Оценка позы камеры с помощью PnP
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=odometry_calculator.camera_matrix,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not retval or inliers is None or len(inliers) < 4:
                logger.warning("Не удалось оценить позу камеры с помощью PnP. Пропуск кадра.")
                continue

            # Преобразуем rvec и tvec в матрицу R и вектор t
            R, _ = cv2.Rodrigues(rvec)
            t = tvec

            # Сохраняем позу камеры
            current_pose = np.hstack((R, t))
            poses.append(current_pose)

            # Добавляем новый keyframe каждые N кадров или по определенным критериям
            if frame_idx % 5 == 0:
                keyframes.append((frame_idx, curr_keypoints, curr_descriptors, current_pose))
                # Обновляем map points с использованием нового keyframe
                # Выполняем триангуляцию между последним keyframe и текущим кадром
                new_map_points = odometry_calculator.triangulate_new_map_points(
                    keyframes[-2],
                    keyframes[-1],
                    poses,
                    feature_matcher
                )
                # Добавляем новые map points
                map_points.extend(new_map_points)

            # Визуализация и другие обработки
            # Здесь можно добавить код для визуализации траектории и карты
            # Например, обновить график траектории или отобразить 3D карту

            # Обновляем ref_keypoints и ref_descriptors для следующего шага
            ref_keypoints = curr_keypoints
            ref_descriptors = curr_descriptors

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
