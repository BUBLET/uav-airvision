import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import os

class TrajectoryVisualizer:
    def __init__(self, gt_csv: str, est_txt: str):
        # Загрузка данных ground-truth
        data = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
        self.gt = data[:, 1:4]  # x, y, z

        # Загрузка данных оцененной траектории
        est = np.loadtxt(est_txt, delimiter=',', skiprows=1)
        self.est = est[:, 1:4]  # первые три колонки

        # Приводим траектории к одному размеру
        if len(self.est) != len(self.gt):
            n = min(len(self.est), len(self.gt))
            self.est = self.est[:n]
            self.gt = self.gt[:n]

    def compute_ate(self):
        # Выравнивание Procrustes
        mtx1, mtx2, disparity = procrustes(self.gt, self.est)
        errors = np.linalg.norm(mtx1 - mtx2, axis=1)
        return errors, 1000 * np.sqrt((errors**2).mean())  # в мм

    def plot_3d(self, ax=None, data=None, label=None):
        """Метод для построения 3D-графика"""
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection':'3d'})
        ax.plot(data[:, 0], data[:, 1], data[:, 2], label=label)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
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
        """Отображение траекторий на 6 отдельных графиках"""
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # 1. Ground Truth - X координата
        axs[0, 0].plot(self.gt[:, 0], label='Ground Truth X', color='b')
        axs[0, 0].set_title('Ground Truth X Coordinate')
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('X (m)')

        # 2. Estimated - X координата
        axs[0, 1].plot(self.est[:, 0], label='Estimated X', color='r')
        axs[0, 1].set_title('Estimated X Coordinate')
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('X (m)')

        # 3. Ground Truth - Y координата
        axs[1, 0].plot(self.gt[:, 1], label='Ground Truth Y', color='b')
        axs[1, 0].set_title('Ground Truth Y Coordinate')
        axs[1, 0].set_xlabel('Frame')
        axs[1, 0].set_ylabel('Y (m)')

        # 4. Estimated - Y координата
        axs[1, 1].plot(self.est[:, 1], label='Estimated Y', color='r')
        axs[1, 1].set_title('Estimated Y Coordinate')
        axs[1, 1].set_xlabel('Frame')
        axs[1, 1].set_ylabel('Y (m)')

        # 5. Ground Truth - Z координата
        axs[2, 0].plot(self.gt[:, 2], label='Ground Truth Z', color='b')
        axs[2, 0].set_title('Ground Truth Z Coordinate')
        axs[2, 0].set_xlabel('Frame')
        axs[2, 0].set_ylabel('Z (m)')

        # 6. Estimated - Z координата
        axs[2, 1].plot(self.est[:, 2], label='Estimated Z', color='r')
        axs[2, 1].set_title('Estimated Z Coordinate')
        axs[2, 1].set_xlabel('Frame')
        axs[2, 1].set_ylabel('Z (m)')

        plt.tight_layout()
        plt.show()
