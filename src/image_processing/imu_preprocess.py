import pandas as pd
import numpy as np
from pathlib import Path


def preprocess_imu(dataset_path: Path, cam_to_imu_timeshift_s: float) -> None:
    """
    Интерполирует данные Ground Truth из state_groundtruth_estimate0 в IMU поток и сохраняет результат.

    Args:
        dataset_path (Path): путь к корню датасета.
        cam_to_imu_timeshift_s (float): временной сдвиг между камерой и IMU в секундах.
    """
    imu_csv = dataset_path / 'imu0' / 'data.csv'
    gt_csv  = dataset_path / 'state_groundtruth_estimate0' / 'data.csv'

    imu_df = pd.read_csv(imu_csv)
    gt_df  = pd.read_csv(gt_csv)

    ts_col = '#timestamp' if '#timestamp' in gt_df.columns else gt_df.columns[0]
    shift_ns = int(cam_to_imu_timeshift_s * 1e9)
    gt_df[ts_col] += shift_ns

    gt_df = gt_df.rename(columns={ts_col: '#timestamp [ns]'})
    gt_df.set_index('#timestamp [ns]', inplace=True)
    gt_df.sort_index(inplace=True)

    for col in gt_df.select_dtypes(include=[np.number]).columns:
        imu_df[col] = np.interp(
            imu_df['#timestamp [ns]'],
            gt_df.index.values,
            gt_df[col].values
        )

    out = dataset_path / 'imu0' / 'imu_with_interpolated_groundtruth.csv'
    imu_df.to_csv(out, index=False)
