from config import DATASET_PATH, IMU_TIMESHIFT_S
from image_processing import (
    PinholeCamera,
    VisualOdometry,
    TrajectoryVisualizer,
    FrameProcessor,
    OdometryPipeline,
    EurocLoader,
    preprocess_imu
)

def build_pipeline():
    # 1) Предобработка IMU
    preprocess_imu(DATASET_PATH, IMU_TIMESHIFT_S)

    # 2) Загрузка данных
    loader = EurocLoader()
    gt_df, cam_df = loader.load_imu(), loader.load_cam()

    # 3) Инициализация компонентов
    camera    = PinholeCamera.from_config()
    vo        = VisualOdometry(camera, gt_df, cam_df)
    viz       = TrajectoryVisualizer(size=400, scale=25)
    processor = FrameProcessor(camera, vo, viz, gt_df)
    pipeline  = OdometryPipeline(
        cam_df=cam_df,
        processor=processor,
        viz_show_func=viz.show,
        viz_save_func=viz.save
    )

    return pipeline

def run_pipeline():
    pipeline = build_pipeline()
    return pipeline.run()
