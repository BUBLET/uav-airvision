import numpy as np
import cv2
from typing import List, Tuple
from .imu_synchronizer import IMUSynchronizer
from .feature_extraction import FeatureExtractor
from .odometry_calculation import OdometryCalculator
from .utils import quat_to_rotmat, normalize_quat
from filterpy.kalman import KalmanFilter

class WindowOptimizer:
    def __init__(self, K: np.ndarray, imu_synchronizer: IMUSynchronizer, window_size: int, dt: float):
        self.K = K
        self.imu = imu_synchronizer
        self.window_size = window_size  # Указываем размер окна
        self.dt = dt
        self.reset()

        # Инициализация Калмановского фильтра для множества состояний
        self.kf = KalmanFilter(dim_x=6 * window_size, dim_z=3 * window_size)  # Поддержка множества состояний
        self._init_ekf()

    def reset(self):
        # Инициализация всех состояний
        self.cam_states = []  # Список для хранения состояний камеры
        self.imu_deltas = []  # Список дельт IMU
        self.observations = []  # Наблюдения
        self.prev_state = None  # Предыдущее состояние
        self.prev_pose = np.zeros(6)  # Начальная поза

    def add_keyframe(self, qvec: np.ndarray, tvec: np.ndarray, bias_gyro: np.ndarray, bias_accel: np.ndarray):
        """
        Добавление нового ключевого кадра с позой и смещением IMU.
        """
        idx = len(self.cam_states)
        keyframe_state = np.hstack((qvec, tvec, bias_gyro, bias_accel))  # Составляем состояние
        self.cam_states.append(keyframe_state)
        # Ограничиваем размер окна
        if len(self.cam_states) > self.window_size:
            self.cam_states.pop(0)  # Удаляем старые состояния, если окно переполнено
        return idx

    def add_imu_prediction(self, imu_data: np.ndarray):
        """
        Применение данных IMU для предсказания состояний.
        """
        for imu_entry in imu_data:
            predicted_state = self.imu.preintegrate(imu_entry[0], imu_entry[1])  # Используем IMU для предсказания
            self.cam_states.append(predicted_state)
            # Ограничиваем размер окна
            if len(self.cam_states) > self.window_size:
                self.cam_states.pop(0)  # Удаляем старые состояния

    def add_observation(self, frame_idx: int, pt_idx: int, uv: np.ndarray):
        """
        Добавление наблюдения для текущего кадра.
        """
        self.observations.append((frame_idx, pt_idx, uv))

    def update_state(self):
        """
        Обновление состояний на основе визуальных наблюдений и IMU.
        """
        for frame_idx, pt_idx, uv in self.observations:
            R_cam = self.cam_states[frame_idx][:3]  # Ориентация камеры для кадра
            t_cam = self.cam_states[frame_idx][3:6]  # Позиция камеры для кадра

            pts_projected = cv2.projectPoints(self.cam_states[pt_idx], R_cam, t_cam, self.K, None)
            error = np.linalg.norm(uv - pts_projected)  # Ошибка репроекции
            if error > 1.0:
                pass  # Логика для коррекции ошибки

    def solve(self):
        """
        Решение проблемы оптимизации с использованием EKF и MSCKF.
        """
        imu_data = self.imu.get_window(self.prev_ts, self.prev_ts + 0.1)  # Получение данных IMU
        self.add_imu_prediction(imu_data)

        # Применение EKF для обновления состояния
        self.kf.predict()

        # Обновление состояния на основе визуальных наблюдений
        self.update_state()

        for frame_idx, pt_idx, uv in self.observations:
            self.kf.update(uv)  # Обновление состояния с использованием визуальных данных
