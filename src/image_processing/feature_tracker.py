import numpy as np
import cv2
from itertools import chain

from .feature_meta_data import FeatureMetaData
from .utils import select

class FeatureTracker:
    def __init__(self,
                 lk_params,
                 imu_processor,
                 stereo_matcher,
                 cam0_intrinsics,
                 cam0_distortion_model,
                 cam0_distortion_coeffs,
                 cam1_intrinsics,
                 cam1_distortion_model,
                 cam1_distortion_coeffs,
                 prev_cam0_pyramid,
                 curr_cam0_pyramid,
                 prev_features,
                 curr_features,
                 num_features,
                 grid_row,
                 grid_col,
                 ransac_threshold):
        """
        lk_params:            dict для cv2.calcOpticalFlowPyrLK
        imu_processor:        IMUProcessor (с методами integrate_imu_data и полями R_cam?_imu)
        stereo_matcher:       StereoMatcher (с методом stereo_match)
        cam?_intrinsics:      vec4 [fx, fy, cx, cy]
        cam?_distortion_*:    модель и коэффициенты дисторсии
        prev_cam0_pyramid:    пирамида предыдущего кадра (левая камера)
        curr_cam0_pyramid:    пирамида текущего кадра (левая камера)
        prev_features:        список списков FeatureMetaData для предыдущего кадра
        curr_features:        список списков FeatureMetaData для заполнения
        num_features:         dict для записи количества точек на этапах
        grid_row/col:         параметры разбиения кадра на сетку
        ransac_threshold:     порог для RANSAC-фильтрации
        """
        self.lk_params            = lk_params
        self.integrate_imu_data   = imu_processor.integrate_imu_data
        self.R_cam0_imu           = imu_processor.R_cam0_imu
        self.R_cam1_imu           = imu_processor.R_cam1_imu
        self.stereo_match         = stereo_matcher.stereo_match

        self.cam0_intrinsics      = cam0_intrinsics
        self.cam0_dist_model      = cam0_distortion_model
        self.cam0_dist_coeffs     = cam0_distortion_coeffs
        self.cam1_intrinsics      = cam1_intrinsics
        self.cam1_dist_model      = cam1_distortion_model
        self.cam1_dist_coeffs     = cam1_distortion_coeffs

        self.prev_cam0_pyramid    = prev_cam0_pyramid
        self.curr_cam0_pyramid    = curr_cam0_pyramid

        self.prev_features        = prev_features
        self.curr_features        = curr_features
        self.num_features         = num_features

        self.grid_row             = grid_row
        self.grid_col             = grid_col
        self.ransac_threshold     = ransac_threshold

    def get_grid_size(self, img):
        """
        Возвращает (height, width) одной ячейки сетки.
        """
        h, w = img.shape[:2]
        grid_h = int(np.ceil(h / self.grid_row))
        grid_w = int(np.ceil(w / self.grid_col))
        return grid_h, grid_w

    def track_features(self):
        """
        Основной трекинг: LK + стерео матчинг + RANSAC + обновление curr_features.
        """
        # 1) Сетка по размеру изображения (из пирамиды берём форму)
        img = self.curr_cam0_pyramid
        grid_h, grid_w = self.get_grid_size(img)

        # 2) Предсказание поворота из IMU
        cam0_R_p_c, cam1_R_p_c = self.integrate_imu_data()

        # 3) Собираем прошлые точки
        prev_ids, prev_lifetime = [], []
        prev_cam0_pts, prev_cam1_pts = [], []
        for f in chain.from_iterable(self.prev_features):
            prev_ids.append(f.id)
            prev_lifetime.append(f.lifetime)
            prev_cam0_pts.append(f.cam0_point)
            prev_cam1_pts.append(f.cam1_point)
        prev_cam0_pts = np.array(prev_cam0_pts, dtype=np.float32)

        # 4) Записываем число до трекинга
        self.num_features['before_tracking'] = len(prev_cam0_pts)
        if len(prev_cam0_pts) == 0:
            return

        # 5) Предсказание положения + LK-tracker
        pred_pts = self.predict_feature_tracking(prev_cam0_pts, cam0_R_p_c, self.cam0_intrinsics)
        curr_pts, track_mask, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_cam0_pyramid,
            self.curr_cam0_pyramid,
            prev_cam0_pts,
            pred_pts,
            **self.lk_params
        )

        # 6) Отсечение выходящих за рамки
        for i, p in enumerate(curr_pts):
            if not track_mask[i]:
                continue
            if p[0] < 0 or p[0] > img.shape[1]-1 or p[1] < 0 or p[1] > img.shape[0]-1:
                track_mask[i] = 0

        # 7) Сбор оттрекиненных точек
        prev_tr_ids    = select(prev_ids, track_mask)
        prev_tr_life   = select(prev_lifetime, track_mask)
        prev_tr_cam0   = select(prev_cam0_pts, track_mask)
        prev_tr_cam1   = select(prev_cam1_pts, track_mask)
        curr_tr_cam0   = select(curr_pts, track_mask)
        self.num_features['after_tracking'] = len(curr_tr_cam0)

        # 8) Стерео матчинг
        curr_cam1_pts, match_mask = self.stereo_match(curr_tr_cam0)
        pm_ids   = select(prev_tr_ids,    match_mask)
        pm_life  = select(prev_tr_life,   match_mask)
        pm_cam0  = select(prev_tr_cam0,   match_mask)
        pm_cam1  = select(prev_tr_cam1,   match_mask)
        cm_cam0  = select(curr_tr_cam0,   match_mask)
        cm_cam1  = select(curr_cam1_pts,  match_mask)
        self.num_features['after_matching'] = len(cm_cam0)

        # 9) RANSAC-шаблон (здесь заглушка — все inliers)
        cam0_inls = [1] * len(pm_cam0)
        cam1_inls = [1] * len(pm_cam1)

        # 10) Обновляем curr_features
        cnt = 0
        for i in range(len(cam0_inls)):
            if not (cam0_inls[i] and cam1_inls[i]):
                continue
            pt = cm_cam0[i]
            row = int(pt[1] / grid_h)
            col = int(pt[0] / grid_w)
            idx = row * self.grid_col + col

            fm = FeatureMetaData()
            fm.id         = pm_ids[i]
            fm.lifetime   = pm_life[i] + 1
            fm.cam0_point = cm_cam0[i]
            fm.cam1_point = cm_cam1[i]

            self.curr_features[idx].append(fm)
            cnt += 1

        self.num_features['after_ransac'] = cnt

    def predict_feature_tracking(self, input_pts, R_p_c, intrinsics):
        """
        Компенсация вращения перед трекингом.
        """
        if len(input_pts) == 0:
            return np.array([], dtype=np.float32)

        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]
        ])
        H = K @ R_p_c @ np.linalg.inv(K)

        pts = []
        for p in input_pts:
            h = H @ np.array([p[0], p[1], 1.0])
            pts.append([h[0]/h[2], h[1]/h[2]])
        return np.array(pts, dtype=np.float32)
