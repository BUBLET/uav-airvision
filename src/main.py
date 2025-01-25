import logging
import cv2
import numpy as np

from image_processing import FeatureExtractor, FeatureMatcher, OdometryCalculator, MapPoint, FrameProcessor
from error_correction.error_correction import ErrorCorrector 
from optimization.ba import BundleAdjustment
# from visualization import Visualizer3D   


def configure_logging():
    """
    Настраивает логгирование для runtime и warning, а также для метрик.
    """
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

    return logger, metrics_logger


def run_pipeline(
    video_path = "D:/Мисис/Диплом/AirVision/datasets/output.mp4",
    lost_threshold: int = 10,
    output_trajectory_path: str = "results/estimated_traj.txt"
):
    """
    Запускает основной процесс VSLAM/одометрии на входном видео.

    :param video_path: Путь до видеофайла.
    :param lost_threshold: Число подряд пропущенных кадров, после которого сбрасываем инициализацию.
    :param output_trajectory_path: Путь для сохранения траектории (txt-файл).
    """
    logger, metrics_logger = configure_logging()
    logger.info(f"Запуск с video_path={video_path}, lost_threshold={lost_threshold}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видео.")
        return

    ret, reference_frame = cap.read()
    if not ret:
        logger.error("Первый кадр отсутствует.")
        return

    frame_height, frame_width = reference_frame.shape[:2]

    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    odometry_calculator = OdometryCalculator(image_width=frame_width, image_height=frame_height)
    
    processor = FrameProcessor(
        feature_extractor=feature_extractor,
        feature_matcher=feature_matcher,
        odometry_calculator=odometry_calculator
    )

    # Если нужно - инициализируем визуализатор
    # visualizer = Visualizer3D()

    frame_idx = 1
    initialization_completed = False
    reset_vis = False
    lost_frames_count = 0

    map_points = []
    keyframes = []
    poses = []

    ref_keypoints, ref_descriptors = feature_extractor.extract_features(reference_frame)
    if not ref_keypoints:
        logger.error("В опорном кадре не удалось найти ключевые точки.")
        return


    initial_translation = np.zeros((3, 1), dtype=np.float32)  
    initial_rotation = np.eye(3, dtype=np.float32)
    initial_pose = np.hstack((initial_rotation, initial_translation))
    keyframes.append((frame_idx, ref_keypoints, ref_descriptors, initial_pose))
    poses.append(initial_pose)
    last_pose = initial_pose

 
    traj_file = open(output_trajectory_path, 'w')
    x, y, z = last_pose[0, 3], last_pose[1, 3], last_pose[2, 3]
    fout_line = f"{x} {y} {z} "
    R = last_pose[:, :3]
    for i in range(3):
        for j in range(3):
            fout_line += f"{R[j, i]} "
    fout_line += "\n"
    traj_file.write(fout_line)

    while True:
        # Читаем следующий кадр
        ret, current_frame = cap.read()
        if not ret:
            logger.info("Достигнут конец видео.")
            break

        frame_idx += 1


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

        # Если result = None, значит кадр не обработан (например, мало фич или нет позы)
        if result is None:
            lost_frames_count += 1
            if lost_frames_count > lost_threshold:
                # Сброс инициализации
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
            R_flat = last_pose[:3, :3].flatten()
            fout_line = f"{x} {y} {z} " + " ".join(map(str, R_flat)) + "\n"
            traj_file.write(fout_line)

        
        # trajectory = [pose[:3, 3] for pose in poses]
        # map_points_coordinates = [mp.coordinates for mp in map_points if mp.coordinates[2] > 0]
        #
        # visualizer.update_trajectory(trajectory)
        # visualizer.update_map_points(map_points_coordinates)
        # visualizer.render()
        #
        # if not reset_vis:
        #     visualizer.vis.reset_view_point(True)
        #     reset_vis = True

    # Освобождаем ресурсы
    # visualizer.close()  # если использовали визуализатор
    traj_file.close()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Обработка завершена.")


def main():

    run_pipeline(
        video_path="datasets/output.mp4",
        lost_threshold=10,
        output_trajectory_path="results/estimated_traj.txt"
    )


if __name__ == "__main__":
    main()
