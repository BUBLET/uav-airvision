import logging
import config
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

runtime_handler = logging.FileHandler("logs/runtime.log", mode='w')
runtime_handler.setLevel(logging.INFO)
runtime_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
runtime_handler.setFormatter(runtime_formatter)
logger.addHandler(runtime_handler)

warning_handler = logging.FileHandler("logs/error.log", mode="w")
warning_handler.setLevel(logging.WARNING)
warning_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
warning_handler.setFormatter(warning_formatter)
logger.addHandler(warning_handler)

# Логгер метрик
metrics_logger = logging.getLogger("metrics_logger")
metrics_logger.setLevel(logging.INFO)
metrics_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
metrics_file_handler = logging.FileHandler("logs/metrics.log", mode='w')
metrics_file_handler.setFormatter(metrics_formatter)
metrics_logger.addHandler(metrics_file_handler)

from image_processing import FeatureExtractor, FeatureMatcher, OdometryCalculator, MapPoint, FrameProcessor
from error_correction.error_correction import ErrorCorrector
from src.optimization.ba import BundleAdjustment
from visualization import Visualizer3D


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
    #visualizer = Visualizer3D()
    
    # Инициализируем переменные 
    frame_idx = 1 # Индекс текущего кадра
    initialization_completed = False # Флаг инициализации
    reset_vis = False
    
    lost_frames_count = 0
    
    # Хранение мап поинтс, кейфрамес и poses
    map_points = []
    keyframes = []
    poses = []

    # Находим кей поинтс и дескрипторы для опорного кадра
    ref_keypoints, ref_descriptors = feature_extractor.extract_features(reference_frame)
    
    if not ref_keypoints:
        logger.error("no keypoints")
        return
    
    # Создаем первый ключевой кадр
    initial_translation = np.zeros((3, 1), dtype=np.float32)  # [0,0,0]
    initial_rotation = np.eye(3, dtype=np.float32)
    initial_pose = np.hstack((initial_rotation, initial_translation))
    keyframes.append((frame_idx, ref_keypoints, ref_descriptors, initial_pose))
    poses.append(initial_pose)
    last_pose = initial_pose

    trajectory_file_path = 'results/estimated_traj.txt'
    traj_file = open(trajectory_file_path, 'w')
    x, y, z = last_pose[0, 3], last_pose[1, 3], last_pose[2, 3]
    fout_line = f"{x} {y} {z} "
    R = last_pose[:, :3]
    for i in range(3):
        for j in range(3):
            fout_line += f"{R[j, i]} "
    fout_line += "\n"
    traj_file.write(fout_line)

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
            lost_frames_count += 1
            if lost_frames_count > config.LOST_THRESHOLD:
                initialization_completed = False
                map_points = []
                keyframes = []
                poses = []
                ref_keypoints = None
                ref_descriptors = None
                metrics_logger.info(f"[LOST] Frame {frame_idx}, lost_frames_count={lost_frames_count}")
            continue
        else:
            lost_frames_count = 0
            ref_keypoints, ref_descriptors, last_pose, map_points, initialization_completed = result
            
            # Запись траектории
            x, y, z = last_pose[:3, 3]
            R = last_pose[:3, :3].flatten()
            fout_line = f"{x} {y} {z} " + " ".join(map(str, R)) + "\n"
            traj_file.write(fout_line)
        # Визуализация
        
    #     trajectory = [pose[:3, 3] for pose in poses]  # Извлекаем центры камер
    #     map_points_coordinates = [mp.coordinates for mp in map_points if mp.coordinates[2] > 0]

    #     if len(trajectory) > 0:
    #         logger.info(f"Trajectory length: {len(trajectory)}, First point: {trajectory[0]}")
    #     else:
    #         logger.warning("Trajectory is empty.")

    #     if len(map_points_coordinates) > 0:
    #         logger.info(f"Map points count: {len(map_points_coordinates)}, First point: {map_points_coordinates[0]}")
    #     else:
    #         logger.warning("Map points are empty.")

    #     visualizer.update_trajectory(trajectory)
    #     visualizer.update_map_points(map_points_coordinates)
    #     visualizer.render()

    #     if not reset_vis:
    #         visualizer.vis.reset_view_point(True)
    #         reset_vis = True

    # # Освобождаем ресурсы
    # visualizer.close()
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
