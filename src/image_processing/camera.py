import cv2
import numpy as np

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
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        return undistorted