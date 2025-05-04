import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Iterator
from config import DATASET_PATH

class EurocLoader:
    """
    Загрузчик датасета EuRoC MAV:
      - imu_df: DataFrame
      - cam_df: DataFrame
      - iter_frames()
    """
    def __init__(self, dataset_path: str = DATASET_PATH, cam_topic: str = 'cam0', imu_topic: str = 'imu0'):
        self.dataset_path = Path(dataset_path)
        self.imu_dir = self.dataset_path / imu_topic
        self.cam_dir = self.dataset_path / cam_topic
        self.imu_csv = self.imu_dir / 'imu_with_interpolated_groundtruth.csv'
        self.cam_csv = self.cam_dir / 'data.csv'
        self.cam_data_dir = self.cam_dir / 'data'

        self.imu_df = None
        self.cam_df = None

    def load_imu(self) -> pd.DataFrame:
        """Загружает IMU+GT CSV и сохраняет в self.imu_df."""
        self.imu_df = pd.read_csv(self.imu_csv)
        return self.imu_df

    def load_cam(self) -> pd.DataFrame:
        """Загружает CSV с таймстампами кадров и сохраняет в self.cam_df."""
        self.cam_df = pd.read_csv(self.cam_csv)
        return self.cam_df

    def iter_frames(self) -> Iterator[Tuple[int, 'np.ndarray']]:
        """
        По всем кадрам:
        Yields:
            ts (int)
            img (np.ndarray)
        """
        if self.cam_df is None:
            self.load_cam()

        for _, row in self.cam_df.iterrows():
            ts = int(row['#timestamp [ns]'])
            img_path = self.cam_data_dir / f"{ts}.png"
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            yield ts, img
