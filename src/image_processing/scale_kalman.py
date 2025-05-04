import cv2
import numpy as np

class ScaleKalmanFilter:
    """
    1D Kalman filter for smoothing scale estimates (distance between frames).
    State vector: [scale, scale_rate]
    Measurement: [scale]
    """
    def __init__(self, init_scale: float = 1.0,
                 init_rate: float = 0.0,
                 process_noise: tuple = (1e-4, 1e-6),
                 measurement_noise: float = 1e-2,
                 init_covariance: float = 0.1):

        self.kf = cv2.KalmanFilter(dynamParams=2, measureParams=1, controlParams=0)

        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)

        self.kf.statePost = np.array([[init_scale], [init_rate]], dtype=np.float32)

        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * init_covariance

        q_scale, q_rate = process_noise
        self.kf.processNoiseCov = np.array([[q_scale,    0.0],
                                            [   0.0, q_rate ]], dtype=np.float32)

        self.kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)

        self.kf.transitionMatrix = np.eye(2, dtype=np.float32)

    def predict(self, dt: float):
        self.kf.transitionMatrix = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        return self.kf.predict()

    def correct(self, raw_scale: float):
        measurement = np.array([[raw_scale]], dtype=np.float32)
        return self.kf.correct(measurement)

    @property
    def scale(self) -> float:
        return float(self.kf.statePost[0, 0])
