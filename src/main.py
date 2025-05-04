# main.py

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import DATASET_PATH, IMU_TIMESHIFT_S, OUTPUT_TRAJ, CAM_PARAMS
from image_processing.visual_odometry import VisualOdometry
from image_processing.utils import rotation_matrix_to_euler

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]
        self.distortion = any(abs(p) > 1e-7 for p in self.d)

    def undistort_image(self, img):
        if not self.distortion:
            return img
        camera_matrix = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0,       0,       1]], dtype=np.float32)
        dist_coeffs = np.array(self.d, dtype=np.float32)
        return cv2.undistort(img, camera_matrix, dist_coeffs)


def preprocess_imu(dataset_path: Path, cam_to_imu_timeshift_s: float):
    imu_csv = dataset_path / 'imu0' / 'data.csv'
    gt_csv  = dataset_path / 'state_groundtruth_estimate0' / 'data.csv'
    imu_df = pd.read_csv(imu_csv)
    gt_df  = pd.read_csv(gt_csv)

    print(f"[DEBUG] GT columns: {gt_df.columns.tolist()}")

    ts_col = '#timestamp' if '#timestamp' in gt_df.columns else gt_df.columns[0]
    shift_ns = int(cam_to_imu_timeshift_s * 1e9)
    gt_df[ts_col] = gt_df[ts_col] + shift_ns
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
    print(f"[INFO] Preprocessed IMU saved to {out}")


def main():
    preprocess_imu(DATASET_PATH, IMU_TIMESHIFT_S)

    gt_df  = pd.read_csv(DATASET_PATH / 'imu0' / 'imu_with_interpolated_groundtruth.csv')
    cam_df = pd.read_csv(DATASET_PATH / 'cam0' / 'data.csv')

    # распакуем параметры камеры
    cam_args = {
        'width':  CAM_PARAMS['width'],
        'height': CAM_PARAMS['height'],
        'fx':     CAM_PARAMS['fx'],
        'fy':     CAM_PARAMS['fy'],
        'cx':     CAM_PARAMS['cx'],
        'cy':     CAM_PARAMS['cy'],
        **CAM_PARAMS['dist']
    }
    cam = PinholeCamera(**cam_args)

    vo  = VisualOdometry(cam, gt_df, cam_df)

    traj = {
        'xy': np.zeros((800, 800, 3), dtype=np.uint8),
        'xz': np.zeros((800, 800, 3), dtype=np.uint8),
        'yz': np.zeros((800, 800, 3), dtype=np.uint8),
    }
    pred_euler, gt_euler = [], []
    pred_vel,   gt_vel   = [], []
    init_pos = None
    center, scale = 400, 50

    for idx, row in cam_df.iterrows():
        ts = row['#timestamp [ns]']
        img_path = DATASET_PATH / 'cam0' / 'data' / f"{ts}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Cannot read {img_path}")
            continue

        img = cam.undistort_image(img)
        vo.update(img, idx)
        if idx <= 2:
            continue

        # VO prediction
        t = vo.cur_t.flatten()
        e = rotation_matrix_to_euler(vo.cur_R)
        v = vo.cur_vel.flatten()
        pred_euler.append(e);  pred_vel.append(v)

        # Ground truth
        row_gt = gt_df[gt_df['#timestamp [ns]'] == ts]
        if not row_gt.empty:
            r = row_gt.iloc[0]
            Rgt = vo.quaternion_to_rotation_matrix(
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
            e_gt = np.zeros(3); v_gt = np.zeros(3)
        gt_euler.append(e_gt);  gt_vel.append(v_gt)

        if init_pos is None:
            init_pos = t.copy()

        def draw(a1, a2, color_pred, color_gt, canvas):
            true_vals = [vo.trueX, vo.trueY, vo.trueZ]

            x1 = int((t[a1] - init_pos[a1]) * scale) + center
            y1 = int((t[a2] - init_pos[a2]) * scale) + center
            x2 = int((true_vals[a1] - init_pos[a1]) * scale) + center
            y2 = int((true_vals[a2] - init_pos[a2]) * scale) + center
            if 0 <= x1 < 800 and 0 <= y1 < 800:
                cv2.circle(canvas, (x1, y1), 1, color_pred, 1)
            if 0 <= x2 < 800 and 0 <= y2 < 800:
                cv2.circle(canvas, (x2, y2), 1, color_gt, 2)

        draw(0, 1, (0, 255, 0), (0, 0, 255), traj['xy'])
        draw(0, 2, (0, 255, 0), (0, 0, 255), traj['xz'])
        draw(1, 2, (0, 255, 0), (0, 0, 255), traj['yz'])

        cv2.imshow('Traj XY', traj['xy'])
        cv2.imshow('Traj XZ', traj['xz'])
        cv2.imshow('Traj YZ', traj['yz'])
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for key, path in OUTPUT_TRAJ.items():
        cv2.imwrite(str(path), traj[key])
        print(f"[INFO] Saved {path}")

    cv2.destroyAllWindows()

    # Графики
    pred_e = np.rad2deg(np.unwrap(np.array(pred_euler), axis=0))
    gt_e   = np.rad2deg(np.unwrap(np.array(gt_euler),   axis=0))
    pred_v = np.array(pred_vel)
    gt_v   = np.array(gt_vel)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    titles_e = ['Roll', 'Pitch', 'Yaw']
    titles_v = ['Vx', 'Vy', 'Vz']
    for i in range(3):
        axs[0, i].plot(pred_e[:, i], label='VO')
        axs[0, i].plot(gt_e[:, i],   label='GT')
        axs[0, i].set_title(f"{titles_e[i]} (deg)")
        axs[0, i].legend(); axs[0, i].grid(True)

        axs[1, i].plot(pred_v[:, i], label='VO')
        axs[1, i].plot(gt_v[:, i],   label='GT')
        axs[1, i].set_title(f"{titles_v[i]} (m/s)")
        axs[1, i].legend(); axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
