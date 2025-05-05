import cv2
import numpy as np
import threading
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleViewer:
    def __init__(self, history=500):
        # буферы
        self.img = None
        self.poses = []  # список [x,y,z]
        self.lock = threading.Lock()
        self.history = history

        # запуск потока отрисовки
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def update_image(self, img):
        # img — BGR numpy array
        with self.lock:
            self.img = img.copy()

    def update_pose(self, T_c_w):
        # T_c_w — Isometry3d: .t (3,), .R (3×3)
        pos = T_c_w.t
        with self.lock:
            self.poses.append(pos.tolist())
            if len(self.poses) > self.history:
                self.poses.pop(0)

    def _run(self):
        # окна
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        plt.ion()
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')

        while self.running:
            time.sleep(1/20)  # 20 FPS redraw

            # копируем под отрисовку
            with self.lock:
                img = self.img.copy() if self.img is not None else None
                poses = np.array(self.poses)

            # показываем кадр
            if img is not None:
                cv2.imshow('frame', img)
            if cv2.waitKey(1) == 27:  # Esc
                self.running = False
                break

            # отрисовка траектории
            if poses.shape[0] > 1:
                ax.clear()
                ax.plot(poses[:,0], poses[:,1], poses[:,2], '-o', markersize=2)
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
                ax.set_title('Trajectory')
                plt.draw()
                plt.pause(0.001)

        cv2.destroyAllWindows()
        plt.close(fig)
