import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from image_processing.camera import PinholeCamera
from image_processing.visual_odometry import VisualOdometry
from image_processing.utils import rotation_matrix_to_euler
from image_processing.visualizer import TrajectoryVisualizer

class FrameProcessor:
    def __init__(
        self,
        camera: PinholeCamera,
        vo: VisualOdometry,
        viz: TrajectoryVisualizer,
        gt_df: pd.DataFrame
    ):
        """
        :param camera: экземпляр PinholeCamera
        :param vo: экземпляр VisualOdometry
        :param viz: экземпляр TrajectoryVisualizer
        :param gt_df: DataFrame с интерполированным GT (столбец '#timestamp [ns]')
        """
        self.camera = camera
        self.vo = vo
        self.viz = viz
        self.gt_df = gt_df

    def process(
        self,
        idx: int,
        timestamp: int,
        img: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Обработать уже загруженный кадр.

        :param idx: индекс кадра в cam_df
        :param timestamp: значение '#timestamp [ns]'
        :param img: загруженный grayscale-кадр
        :return: (frame, pred_state, gt_state, pred_pos, gt_pos) или None, если кадр пропущен
        """
        # 1) Коррекция дисторсии
        undistorted = self.camera.undistort_image(img)

        # 2) Обновление VO
        self.vo.update(undistorted, idx)
        if idx <= 2:
            return None

        # 3) Собираем предсказанные состояния
        t = self.vo.cur_t.flatten()
        e = rotation_matrix_to_euler(self.vo.cur_R)
        v = self.vo.cur_vel.flatten()
        pred_state = np.concatenate([t, e, v])

        # 4) Собираем GT-состояния
        row_gt = self.gt_df[self.gt_df['#timestamp [ns]'] == timestamp]
        if not row_gt.empty:
            r = row_gt.iloc[0]
            Rgt = self.vo.quaternion_to_rotation_matrix(
                r[' q_RS_w []'], r[' q_RS_x []'],
                r[' q_RS_y []'], r[' q_RS_z []']
            )
            e_gt = rotation_matrix_to_euler(Rgt)
            v_gt = np.array([
                r[' v_RS_R_x [m s^-1]'],
                r[' v_RS_R_y [m s^-1]'],
                r[' v_RS_R_z [m s^-1]']
            ])
        else:
            e_gt = np.zeros(3)
            v_gt = np.zeros(3)
        gt_state = np.concatenate([e_gt, v_gt])

        # 5) Обновление визуализатора и сбор итогового кадра
        self.viz.update(t, [self.vo.trueX, self.vo.trueY, self.vo.trueZ])
        frame = self.viz.compose_frame(undistorted)

        # 6) Позиции
        pred_pos = t.copy()
        gt_pos = np.array([self.vo.trueX, self.vo.trueY, self.vo.trueZ])

        return frame, pred_state, gt_state, pred_pos, gt_pos
