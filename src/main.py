import cv2
import numpy as np
import logging
import config
from image_processing import FeatureExtractor, FeatureMatcher, OdometryCalculator, FrameProcessor
from error_correction.error_correction import ErrorCorrector
from optimization import BundleAdjustment
from visualization import Visualizer3D


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    # Открываем видео
    cap = cv2.VideoCapture(config.VIDEO_PATH)
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

    # Инициализируем обработчик кадров
    processor = FrameProcessor(feature_extractor, feature_matcher, odometry_calculator)
    
    # Инициализация визуализатора
    visualizer = Visualizer3D()
    
    # Инициализируем переменные 
    frame_idx = 1 # Индекс текущего кадра
    initialization_completed = False # Флаг инициализации

    
    # Хранение мап поинтс, кейфрамес и poses
    map_points = []
    keyframes = []
    poses = []

    # Находим кей поинтс и дескрипторы для опорного кадра
    ref_keypoints, ref_descriptors = feature_extractor.extract_features(reference_frame)
    if len(ref_keypoints) == 0:
        logger.error("no keypoints")
        return
    # Создаем первый ключевой кадр
    initial_pose = np.hstack((np.eye(3), np.zeros((3,1))))
    keyframes.append((frame_idx, ref_keypoints, ref_descriptors, initial_pose))
    poses.append(initial_pose)
    last_pose = initial_pose

    while True:
        # Читаем следующий кадр из видео
        ret, current_frame = cap.read()
        if not ret:
            logger.info("complete")
            break
        
        frame_idx += 1

        # Обрабатываем кадр

        result = processor.process_frame(
            frame_idx,
            current_frame,
            ref_keypoints,
            ref_descriptors,
            last_pose,
            map_points,
            initialization_completed,
            poses,
            keyframes
        )

        if result is None:
            continue
        else:
            ref_keypoints, ref_descriptors, last_pose, map_points, initialization_completed = result

        # Визуализация
        
        trajectory = [pose[:3, 3] for pose in poses]  # Извлекаем центры камер
        map_points_coordinates = [mp.coordinates for mp in map_points if mp.coordinates[2] > 0]

        if len(trajectory) > 0:
            logger.info(f"Trajectory length: {len(trajectory)}, First point: {trajectory[0]}")
        else:
            logger.warning("Trajectory is empty.")

        if len(map_points_coordinates) > 0:
            logger.info(f"Map points count: {len(map_points_coordinates)}, First point: {map_points_coordinates[0]}")
        else:
            logger.warning("Map points are empty.")

        visualizer.update_trajectory(trajectory)
        visualizer.update_map_points(map_points_coordinates)
        visualizer.render()

    # Освобождаем ресурсы
    visualizer.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
