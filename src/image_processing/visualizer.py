import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

class TrajectoryVisualizer:
    def __init__(self, gt_csv: str, est_txt: str):
        # GT
        data_gt = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
        gt = data_gt[:, 1:4]  # x,y,z
        # Est
        data_est = np.loadtxt(est_txt, delimiter=',', skiprows=1)
        est = data_est[:, 1:4]

        # Обрезаем до одинаковой длинны
        n = min(len(gt), len(est))
        gt = gt[:n]
        est = est[:n]

        # Сдвигаем так, чтобы обе начинались в начале координат
        self.gt = gt - gt[0]
        self.est = est - est[0]

    def compute_ate(self):
        m1, m2, _ = procrustes(self.gt, self.est)
        errors = np.linalg.norm(m1 - m2, axis=1)
        rms = np.sqrt((errors**2).mean())
        return errors, rms

    def plot_3d(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.gt[:,0], self.gt[:,1], self.gt[:,2], label='GT')
        ax.plot(self.est[:,0], self.est[:,1], self.est[:,2], label='Est')

        # Устанавливаем равный масштаб по всем осям
        all_pts = np.vstack((self.gt, self.est))
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
        mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()

    def plot_errors(self):
        errors, rms = self.compute_ate()
        plt.figure(figsize=(6,3))
        plt.plot(errors)
        plt.title(f'ATE per frame (RMS = {rms:.4f} m)')
        plt.xlabel('Frame index')
        plt.ylabel('Error (m)')

    def show_all(self):
        self.plot_3d()
        self.plot_errors()
        plt.tight_layout()
        plt.show()
