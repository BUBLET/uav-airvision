import numpy as np
import cv2

class IMUProcessor:
    def __init__(self, T_imu_cam0, T_imu_cam1):
        """
        T_imu_cam0: 4×4 матрица преобразования из системы IMU в систему камеры-0
        T_imu_cam1: 4×4 матрица преобразования из системы IMU в систему камеры-1
        """
        # compute transforms from camera frames into IMU frame
        self.T_cam0_imu = np.linalg.inv(T_imu_cam0)
        self.R_cam0_imu = self.T_cam0_imu[:3, :3]
        self.t_cam0_imu = self.T_cam0_imu[:3, 3]

        self.T_cam1_imu = np.linalg.inv(T_imu_cam1)
        self.R_cam1_imu = self.T_cam1_imu[:3, :3]
        self.t_cam1_imu = self.T_cam1_imu[:3, 3]

        # buffer for incoming IMU messages
        self.imu_buffer = []

    def imu_callback(self, msg):
        """
        Колбэк для входящих сообщений IMU.
        """
        self.imu_buffer.append(msg)

    def integrate_imu_data(self):
        """
        Интегрирует гироскопические данные IMU между двумя последовательными изображениями,
        используя временные метки self.cam0_prev_img_msg и self.cam0_curr_img_msg.
        
        Возвращает:
            cam0_R_p_c: матрица вращения из предыдущего кадра cam0 в текущий
            cam1_R_p_c: матрица вращения из предыдущего кадра cam1 в текущий
        """
        idx_begin = None
        for i, msg in enumerate(self.imu_buffer):
            if msg.timestamp >= self.cam0_prev_img_msg.timestamp - 0.01:
                idx_begin = i
                break

        idx_end = None
        for i, msg in enumerate(self.imu_buffer):
            if msg.timestamp >= self.cam0_curr_img_msg.timestamp - 0.004:
                idx_end = i
                break

        if idx_begin is None or idx_end is None:
            return np.identity(3), np.identity(3)

        mean_ang_vel = np.zeros(3)
        for msg in self.imu_buffer[idx_begin:idx_end]:
            mean_ang_vel += msg.angular_velocity
        count = idx_end - idx_begin
        if count > 0:
            mean_ang_vel /= count

        cam0_mean = self.R_cam0_imu.T @ mean_ang_vel
        cam1_mean = self.R_cam1_imu.T @ mean_ang_vel

        dt = self.cam0_curr_img_msg.timestamp - self.cam0_prev_img_msg.timestamp
        cam0_R_p_c = cv2.Rodrigues(cam0_mean * dt)[0].T
        cam1_R_p_c = cv2.Rodrigues(cam1_mean * dt)[0].T

        self.imu_buffer = self.imu_buffer[idx_end:]
        return cam0_R_p_c, cam1_R_p_c
