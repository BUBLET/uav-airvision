import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import os

class TrajectoryVisualizer:
    def __init__(self, gt_csv: str, est_txt: str):
        # gt_csv: mav0/state_groundtruth_estimate0/data.csv (timestamp, x,y,z, qx,qy,qz,qw)
        data = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
        self.gt = data[:, 1:4]  # x,y,z

        # est_txt: results/estimated_traj.txt (t_x, t_y, t_z, then R flattened)
        est = np.loadtxt(est_txt, delimiter=',', skiprows=1)
        self.est = est[:, 1:4]    # первые три колонки

        if len(self.est) != len(self.gt):
            n = min(len(self.est), len(self.gt))
            self.est = self.est[:n]
            self.gt = self.gt[:n]

    def compute_ate(self):
        # выравнивание Procrustes
        mtx1, mtx2, disparity = procrustes(self.gt, self.est)
        errors = np.linalg.norm(mtx1 - mtx2, axis=1)
        return errors, 1000 * np.sqrt((errors**2).mean())  # в мм

    def plot_3d(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection':'3d'})
        ax.plot(self.gt[:,0], self.gt[:,1], self.gt[:,2], label='GT')
        ax.plot(self.est[:,0], self.est[:,1], self.est[:,2], label='Est')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        return ax

    def plot_errors(self):
        errors, ate = self.compute_ate()
        plt.figure()
        plt.plot(errors)
        plt.title(f'ATE per frame (RMS = {ate:.2f} mm)')
        plt.xlabel('Frame idx')
        plt.ylabel('Error (m)')
        return errors

    def show_all(self):
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121, projection='3d')
        self.plot_3d(ax)
        plt.subplot(122)
        self.plot_errors()
        plt.tight_layout()
        plt.show()
