import argparse
import logging
import os
import cv2
import numpy as np
from typing import Tuple, Optional

import config
from image_processing import (
    FeatureExtractor,
    OdometryCalculator,
    FrameProcessor,
    TrajectoryWriter
)
from image_processing.odometry_pipeline import OdometryPipeline
from image_processing.dataset_loader import (
    VideoLoader,
    FramesLoader,
    TUMVILoader,
    EuRoCLoader
)
from image_processing.imu_synchronizer import IMUSynchronizer


def configure_logging() -> Tuple[logging.Logger, logging.Logger]:
    logger = logging.getLogger("OdometryPipeline")
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)

    # Основной лог
    runtime_handler = logging.FileHandler("logs/runtime.log", mode='w')
    runtime_handler.setLevel(logging.INFO)
    runtime_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    runtime_handler.setFormatter(runtime_formatter)
    logger.addHandler(runtime_handler)

    # Лог ошибок
    warning_handler = logging.FileHandler("logs/error.log", mode="w")
    warning_handler.setLevel(logging.WARNING)
    warning_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    warning_handler.setFormatter(warning_formatter)
    logger.addHandler(warning_handler)

    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Лог метрик
    metrics_logger = logging.getLogger("metrics_logger")
    metrics_logger.setLevel(logging.INFO)
    metrics_file = logging.FileHandler("logs/metrics.log", mode='w')
    metrics_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    metrics_logger.addHandler(metrics_file)

    return logger, metrics_logger


def initialize_components(frame_width: int, frame_height: int) -> Tuple[FeatureExtractor, OdometryCalculator, FrameProcessor]:
    feature_extractor = FeatureExtractor(
        nfeatures=config.N_FEATURES,
        scaleFactor=config.INIT_SCALE,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=config.WTA_K,
        scoreType=config.SCORE_TYPE,
        patchSize=config.PATCH_SIZE,
        fastThreshold=config.FAST_THRESHOLD
    )

    odometry_calculator = OdometryCalculator(
        image_width=frame_width,
        image_height=frame_height,
        camera_matrix=config.CAMERA_MATRIX,
        e_ransac_threshold=config.E_RANSAC_THRESHOLD
    )

    lk_params = dict(
        winSize=config.LK_WIN_SIZE,
        criteria=config.LK_CRITERIA
    )

    frame_processor = FrameProcessor(feature_extractor, odometry_calculator, lk_params)
    return feature_extractor, odometry_calculator, frame_processor


def make_loader(source_type: str, data_path: str):
    if source_type == "video":
        return VideoLoader(data_path)
    elif source_type == "frames":
        return FramesLoader(data_path, ext="png")
    elif source_type == "tumvi":
        return TUMVILoader(data_path, cam="cam0")
    elif source_type == "euroc":
        return EuRoCLoader(data_path, cam="cam0")
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def main():
    parser = argparse.ArgumentParser(description="Запуск визуальной одометрии/VIO")
    parser.add_argument(
        "--source-type",
        choices=["video", "frames", "tumvi", "euroc"],
        default="video",
        help="Источник данных (по умолчанию: video)"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Путь к видео, папке кадров или корню датасета"
    )
    args = parser.parse_args()

    # Определяем data_path
    if args.data_path is None:
        if args.source_type == "video":
            data_path = config.VIDEO_PATH
        elif args.source_type == "frames":
            data_path = config.FRAMES_FOLDER
        else:
            raise ValueError("Для tumvi/euroc обязательно указать --data-path")
    else:
        data_path = args.data_path

    logger, _ = configure_logging()
    logger.info(f"Запуск с source-type={args.source_type}, data-path={data_path}")

    # Создаём загрузчик изображений
    loader = make_loader(args.source_type, data_path)

    # Создаём синхронизатор IMU (только для TUMVI/EuRoC)
    imu_sync = None
    if args.source_type in ("tumvi", "euroc"):
        imu_csv = os.path.join(data_path, "mav0", "imu0", "data.csv")
        imu_sync = IMUSynchronizer(imu_csv)
        logger.info(f"IMUSynchronizer инициализирован из {imu_csv}")

    # Читаем первый кадр
    first_item = loader.read_next()
    if first_item is None:
        logger.error("Не удалось получить первый кадр.")
        loader.release()
        return
    first_frame, first_ts = first_item

    h, w = first_frame.shape[:2]
    feature_extractor, odometry_calculator, frame_processor = initialize_components(w, h)
    trajectory_writer = TrajectoryWriter("results/estimated_traj.txt")
    pipeline = OdometryPipeline(
        feature_extractor,
        odometry_calculator,
        frame_processor,
        logger,
        trajectory_writer
    )

    # Обрабатываем первый кадр
    init_res = frame_processor.process_frame(first_frame)
    if init_res is None:
        logger.error("Ошибка обработки первого кадра.")
        loader.release()
        trajectory_writer.close()
        return
    _, init_pts, _ = init_res
    pipeline.window_poses.append((pipeline.R_total.copy(), pipeline.t_total.copy()))
    trajectory_writer.write_pose(pipeline.t_total, pipeline.R_total)

    frame_idx = 1
    while True:
        item = loader.read_next()
        if item is None:
            logger.info("Данные кончились.")
            break

        frame, ts = item
        frame_idx += 1

        # Синхронизируем IMU
        if imu_sync is not None:
            gyro, accel = imu_sync.get_measurements_for_frame(ts)
            logger.debug(f"IMU @ {ts:.6f}s — gyro={gyro}, accel={accel}")

        res = frame_processor.process_frame(frame)
        if res is None:
            logger.warning(f"Кадр {frame_idx} пропущен.")
            continue
        _, _, (R_rel, t_rel) = res

        # Обновляем глобальную позу
        pipeline.t_total = pipeline.t_total + pipeline.R_total @ t_rel
        pipeline.R_total = R_rel @ pipeline.R_total

        pipeline.window_relatives.append((R_rel.copy(), t_rel.copy()))
        pipeline.window_poses.append((pipeline.R_total.copy(), pipeline.t_total.copy()))

        if len(pipeline.window_poses) > pipeline.window_size:
            pipeline.optimize_window()
            pipeline.window_poses = [(pipeline.R_total.copy(), pipeline.t_total.copy())]
            pipeline.window_relatives = []

        logger.info(f"Кадр {frame_idx}: t_total={pipeline.t_total.flatten()}")
        trajectory_writer.write_pose(pipeline.t_total, pipeline.R_total)

    loader.release()
    trajectory_writer.close()
    logger.info("Обработка завершена.")


if __name__ == "__main__":
    main()
