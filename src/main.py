#!/usr/bin/env python3

import argparse, glob, logging, os, cv2
from typing import Tuple

import config
from image_processing import (
    FeatureExtractor,
    OdometryCalculator,
    FrameProcessor,
    TrajectoryWriter
)
from image_processing.odometry_pipeline import OdometryPipeline
from image_processing.dataset_loader import VideoLoader, FramesLoader, TUMVILoader, EuRoCLoader
from image_processing.imu_synchronizer import IMUSynchronizer


def configure_logging() -> Tuple[logging.Logger, logging.Logger]:
    logger = logging.getLogger("OdometryPipeline")
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)

    # Основной лог
    rh = logging.FileHandler("logs/runtime.log", mode='w')
    rh.setLevel(logging.INFO)
    rh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(rh)

    # Лог ошибок
    eh = logging.FileHandler("logs/error.log", mode='w')
    eh.setLevel(logging.WARNING)
    eh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(eh)

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # Лог метрик
    metrics = logging.getLogger("metrics_logger")
    metrics.setLevel(logging.INFO)
    mf = logging.FileHandler("logs/metrics.log", mode='w')
    mf.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    metrics.addHandler(mf)

    return logger, metrics


def initialize_components(w: int, h: int, imu):
    fe = FeatureExtractor(
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
    odom = OdometryCalculator(
        image_width=w,
        image_height=h,
        camera_matrix=config.CAMERA_MATRIX
    )
    lk = dict(winSize=config.LK_WIN_SIZE, criteria=config.LK_CRITERIA)
    fp = FrameProcessor(fe, odom, lk, imu_synchronizer=imu)
    return fe, odom, fp


def make_loader(src: str, path: str):
    if src=="video": return VideoLoader(path)
    if src=="frames":return FramesLoader(path,"png")
    if src=="tumvi": return TUMVILoader(path,"cam0")
    if src=="euroc": return EuRoCLoader(path,"cam0")
    raise ValueError(src)


def get_frame_dimensions(source_type: str, data_path: str) -> Tuple[int, int]:
    """
    Возвращает (height, width) первого кадра, не сдвигая основной loader.
    """
    if source_type == "video":
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {data_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр из видео")
    elif source_type == "frames":
        paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
        if not paths:
            raise ValueError(f"Нет файлов PNG в {data_path}")
        frame = cv2.imread(paths[0])
        if frame is None:
            raise ValueError(f"Не удалось прочитать {paths[0]}")
    else:
        # TUMVI и EuRoC: используем временный loader
        temp = make_loader(source_type, data_path)
        item = temp.read_next()
        temp.release()
        if item is None:
            raise ValueError("Не удалось получить первый кадр из датасета")
        frame, _ = item

    return frame.shape[:2]  # (height, width)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-type",choices=["video","frames","tumvi","euroc"],default="video")
    p.add_argument("--data-path",default=None)
    args = p.parse_args()

    if args.data_path is None:
        if args.source_type=="video": data_path = config.VIDEO_PATH
        elif args.source_type=="frames":data_path = config.FRAMES_FOLDER
        else: raise ValueError("need --data-path")
    else:
        data_path = args.data_path

    logger, _ = configure_logging()
    logger.info(f"start {args.source_type} @ {data_path}")

    imu = None
    if args.source_type in ("tumvi","euroc"):
        imu_csv = os.path.join(data_path,"mav0","imu0","data.csv")
        imu = IMUSynchronizer(imu_csv)
        logger.info(f"IMU @ {imu_csv}")

    h,w = get_frame_dimensions(args.source_type,data_path)
    fe, odom, fp = initialize_components(w, h, imu)

    writer   = TrajectoryWriter("results/estimated_traj.txt")
    pipeline = OdometryPipeline(
        fe, odom, fp, imu, logger, writer,
        window_size=config.WINDOW_SIZE,
        T_BS=config.T_BS
    )

    loader = make_loader(args.source_type, data_path)
    pipeline.run(loader)
    loader.release()
    logger.info("done.")


if __name__=="__main__":
    main()