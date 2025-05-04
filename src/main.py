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
        self.width = width;  self.height = height
        self.fx = fx;        self.fy = fy
        self.cx = cx;        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]
        self.distortion = any(abs(p) > 1e-7 for p in self.d)

    def undistort_image(self, img):
        if not self.distortion:
            return img
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0,       0,       1]], dtype=np.float32)
        D = np.array(self.d, dtype=np.float32)
        return cv2.undistort(img, K, D)


def preprocess_imu(dataset_path: Path, cam_to_imu_timeshift_s: float):
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


def main():
    # Параметры визуализации
    VIS_SIZE = 400           # размер каждой ячейки (px)
    center   = VIS_SIZE // 2
    scale    = 25            # уменьшенный масштаб для траекторий

    # 1) Предобработка IMU
    preprocess_imu(DATASET_PATH, IMU_TIMESHIFT_S)

    # 2) Загрузка данных
    gt_df  = pd.read_csv(DATASET_PATH / 'imu0' / 'imu_with_interpolated_groundtruth.csv')
    cam_df = pd.read_csv(DATASET_PATH / 'cam0' / 'data.csv')

    # 3) Инициализация камеры и VO
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

    # 4) Инициализация холстов траекторий размером VIS_SIZE×VIS_SIZE
    traj = {
        'xy': np.zeros((VIS_SIZE, VIS_SIZE, 3), dtype=np.uint8),
        'xz': np.zeros((VIS_SIZE, VIS_SIZE, 3), dtype=np.uint8),
        'yz': np.zeros((VIS_SIZE, VIS_SIZE, 3), dtype=np.uint8),
    }

    pred_euler, gt_euler = [], []
    pred_vel,   gt_vel   = [], []
    init_pos = None

    # 5) Основной цикл
    for idx, row in cam_df.iterrows():
        ts = row['#timestamp [ns]']
        img_path = DATASET_PATH / 'cam0' / 'data' / f"{ts}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 5.1) Коррекция дисторсии
        img = cam.undistort_image(img)

        # 5.2) Обновление VO
        vo.update(img, idx)
        if idx <= 2:
            continue

        # 5.3) Сохранение предсказаний
        t = vo.cur_t.flatten()
        e = rotation_matrix_to_euler(vo.cur_R)
        v = vo.cur_vel.flatten()
        pred_euler.append(e)
        pred_vel.append(v)

        # 5.4) Сохранение GT
        row_gt = gt_df[gt_df['#timestamp [ns]'] == ts]
        if not row_gt.empty:
            r    = row_gt.iloc[0]
            Rgt  = vo.quaternion_to_rotation_matrix(
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
        gt_euler.append(e_gt)
        gt_vel.append(v_gt)

        # 5.5) Установка начальной позиции
        if init_pos is None:
            init_pos = t.copy()

        # 5.6) Функция отрисовки одной проекции
        def draw(a1, a2, canvas):
            true_vals = [vo.trueX, vo.trueY, vo.trueZ]
            x1 = int((t[a1]        - init_pos[a1]) * scale) + center
            y1 = int((t[a2]        - init_pos[a2]) * scale) + center
            x2 = int((true_vals[a1] - init_pos[a1]) * scale) + center
            y2 = int((true_vals[a2] - init_pos[a2]) * scale) + center
            if 0 <= x1 < VIS_SIZE and 0 <= y1 < VIS_SIZE:
                cv2.circle(canvas, (x1, y1), 1, (0,255,0), 1)
            if 0 <= x2 < VIS_SIZE and 0 <= y2 < VIS_SIZE:
                cv2.circle(canvas, (x2, y2), 1, (0,0,255), 2)

        # 5.7) Отрисовка трёх проекций
        draw(0, 1, traj['xy'])
        draw(0, 2, traj['xz'])
        draw(1, 2, traj['yz'])

        # 5.8) Подготовка изображения камеры под размер VIS_SIZE×VIS_SIZE
        cam_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cam_vis = cv2.resize(cam_vis, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_LINEAR)

        # 5.9) Сборка 2×2 доски и отображение
        top    = np.hstack((traj['xy'], traj['xz']))      # VIS_SIZE × (2·VIS_SIZE)
        bottom = np.hstack((traj['yz'], cam_vis))
        board  = np.vstack((top, bottom))                 # (2·VIS_SIZE) × (2·VIS_SIZE)

        cv2.imshow('VO + GT Trajectories', board)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6) Сохранение финальных карт
    for key, path in OUTPUT_TRAJ.items():
        cv2.imwrite(str(path), traj[key])

    cv2.destroyAllWindows()

    # 7) Статические графики углов и скоростей
    pred_e = np.rad2deg(np.unwrap(np.array(pred_euler), axis=0))
    gt_e   = np.rad2deg(np.unwrap(np.array(gt_euler),   axis=0))
    pred_v = np.array(pred_vel)
    gt_v   = np.array(gt_vel)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    titles_e = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    titles_v = ['Vx (m/s)',    'Vy (m/s)',    'Vz (m/s)']

    for i in range(3):
        axs[0, i].plot(pred_e[:, i], label='VO')
        axs[0, i].plot(gt_e[:, i],   label='GT')
        axs[0, i].set_title(titles_e[i])
        axs[0, i].legend()
        axs[0, i].grid(True)

        axs[1, i].plot(pred_v[:, i], label='VO')
        axs[1, i].plot(gt_v[:, i],   label='GT')
        axs[1, i].set_title(titles_v[i])
        axs[1, i].legend()
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
