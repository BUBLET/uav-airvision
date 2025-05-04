import cv2
import numpy as np
from config import CAM_PARAMS

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        # grab dem cam specs
        self.width = width
        self.height = height
        self.fx = fx      # focal length x
        self.fy = fy      # focal length y
        self.cx = cx      # optical center x
        self.cy = cy      # optical center y
        self.d = [k1, k2, p1, p2, k3]
        self.distortion = any(abs(p) > 1e-7 for p in self.d)

    @classmethod
    def from_config(cls):
        """Make a camera from CAM_PARAMS dict."""
        dist = CAM_PARAMS.get('dist', {})
        return cls(
            width=CAM_PARAMS['width'],
            height=CAM_PARAMS['height'],
            fx=CAM_PARAMS['fx'],
            fy=CAM_PARAMS['fy'],
            cx=CAM_PARAMS['cx'],
            cy=CAM_PARAMS['cy'],
            k1=dist.get('k1', 0.0),
            k2=dist.get('k2', 0.0),
            p1=dist.get('p1', 0.0),
            p2=dist.get('p2', 0.0),
            k3=dist.get('k3', 0.0),
        )

    def undistort_image(self, img):
        if not self.distortion:
            return img
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0,       0,       1]], dtype=np.float32)
        D = np.array(self.d, dtype=np.float32)
        return cv2.undistort(img, K, D)
