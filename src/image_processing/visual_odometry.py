# src/image_processing/visual_odometry.py

import numpy as np
import pandas as pd
import cv2
from .constants import STAGE_FIRST_FRAME, STAGE_SECOND_FRAME, STAGE_DEFAULT_FRAME
from .tracker import FeatureTracker
from .utils import rotation_matrix_to_euler, euler_to_rotation_matrix, clamp_euler
from config import VO_PARAMS

class VisualOdometry:
    def __init__(self, cam, gt_data: pd.DataFrame, cam_data: pd.DataFrame):
        self.frame_stage = STAGE_FIRST_FRAME
        self.cam = cam
        self.gt_data  = gt_data.reset_index(drop=True)
        self.cam_data = cam_data.reset_index(drop=True)
        self.tracker = FeatureTracker()

        # инициализация состояния из первого GT
        first = self.gt_data.iloc[0]
        self.cur_t = np.array([first[' p_RS_R_x [m]'],
                               first[' p_RS_R_y [m]'],
                               first[' p_RS_R_z [m]']]).reshape(3,1)
        R_imu = self.quaternion_to_rotation_matrix(
            first[' q_RS_w []'], first[' q_RS_x []'],
            first[' q_RS_y []'], first[' q_RS_z []']
        )
        self.cur_R     = R_imu.copy()
        self.prev_euler = rotation_matrix_to_euler(R_imu)
        self.cur_vel   = np.array([first[' v_RS_R_x [m s^-1]'],
                                   first[' v_RS_R_y [m s^-1]'],
                                   first[' v_RS_R_z [m s^-1]']]).reshape(3,1)

        self.px_ref = None
        self.last_frame, self.new_frame = None, None
        self.focal = cam.fx
        self.pp    = (cam.cx, cam.cy)
        self.T_BS  = np.array(VO_PARAMS["T_BS"])
        self.min_feat = VO_PARAMS["min_features"]
        self.ess_thresh = VO_PARAMS["ess_threshold"]
        self.ess_prob   = VO_PARAMS["ess_prob"]
        self.clamp_deg  = VO_PARAMS["clamp_deg"]

    def quaternion_to_rotation_matrix(self, w, x, y, z):
            R = np.zeros((3,3))
            R[0,0] = 1 - 2*(y*y + z*z)
            R[0,1] = 2*(x*y - z*w)
            R[0,2] = 2*(x*z + y*w)

            R[1,0] = 2*(x*y + z*w)
            R[1,1] = 1 - 2*(x*x + z*z)
            R[1,2] = 2*(y*z - x*w)

            R[2,0] = 2*(x*z - y*w)
            R[2,1] = 2*(y*z + x*w)
            R[2,2] = 1 - 2*(x*x + y*y)
            return R

    def getAbsoluteScale(self, frame_id):
            if frame_id < 1:
                return 0
            curr_timestamp = self.cam_data.iloc[frame_id]['#timestamp [ns]']
            prev_timestamp = self.cam_data.iloc[frame_id - 1]['#timestamp [ns]']
            
            curr_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == curr_timestamp]
            prev_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == prev_timestamp]

            if len(curr_gt) == 0 or len(prev_gt) == 0:
                return 0
            
            curr_gt = curr_gt.iloc[0]
            prev_gt = prev_gt.iloc[0]

            x_prev = prev_gt[' p_RS_R_x [m]']
            y_prev = prev_gt[' p_RS_R_y [m]']
            z_prev = prev_gt[' p_RS_R_z [m]']
            
            x = curr_gt[' p_RS_R_x [m]']
            y = curr_gt[' p_RS_R_y [m]']
            z = curr_gt[' p_RS_R_z [m]']

            self.trueX, self.trueY, self.trueZ = x, y, z
            scale = np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)
            return scale

    def process_first(self):
        self.px_ref = self.tracker.detect(self.new_frame)
        self.frame_stage = STAGE_SECOND_FRAME

    def process_second(self):
        self.px_ref, self.px_cur = self.tracker.track(
            self.last_frame, self.new_frame, self.px_ref
        )
        self._pose_estimation(scale=1.0, threshold=0.1)
        self.frame_stage = STAGE_DEFAULT_FRAME

    def process_default(self, idx):
        self.px_ref, self.px_cur = self.tracker.track(
            self.last_frame, self.new_frame, self.px_ref
        )
        scale = self.getAbsoluteScale(idx)
        if scale > 1e-3:
            self._pose_estimation(scale, idx)
        if len(self.px_ref) < self.min_feat:
            self.px_ref = self.tracker.detect(self.new_frame)

    def _pose_estimation(self, scale, idx=None, threshold=None):
        thr = threshold if threshold is not None else self.ess_thresh
        E, _ = cv2.findEssentialMat(
            self.px_cur, self.px_ref,
            focal=self.focal, pp=self.pp,
            method=cv2.RANSAC, prob=self.ess_prob,
            threshold=thr
        )
        _, R, t, _ = cv2.recoverPose(
            E, self.px_cur, self.px_ref,
            focal=self.focal, pp=self.pp
        )
        T_cam = np.eye(4); T_cam[:3,:3] = R; T_cam[:3,3] = (scale * t).ravel()
        # body-sensor корректировка
        TB_inv = np.linalg.inv(self.T_BS)
        T_corr = self.T_BS @ T_cam @ TB_inv
        R_cam, t_cam = T_corr[:3,:3], T_corr[:3,3].reshape(3,1)

        new_R = self.cur_R @ R_cam
        euler = rotation_matrix_to_euler(new_R)
        clamped = clamp_euler(self.prev_euler, euler, self.clamp_deg)
        self.cur_R     = euler_to_rotation_matrix(clamped)
        self.prev_euler = clamped
        if idx is not None:
            self.cur_t += self.cur_R @ t_cam
            # обновить скорость из GT
            ts = self.cam_data.iloc[idx]['#timestamp [ns]']
            row = self.gt_data[self.gt_data['#timestamp [ns]'] == ts]
            if not row.empty:
                v = row.iloc[0]
                self.cur_vel = np.array([
                    v[' v_RS_R_x [m s^-1]'], v[' v_RS_R_y [m s^-1]'],
                    v[' v_RS_R_z [m s^-1]']
                ]).reshape(3,1)

    def update(self, img, idx):
        assert img.ndim==2, "Только grayscale."
        self.new_frame = img
        if   self.frame_stage == STAGE_FIRST_FRAME:   self.process_first()
        elif self.frame_stage == STAGE_SECOND_FRAME:  self.process_second()
        else:                                         self.process_default(idx)
        self.last_frame = self.new_frame
