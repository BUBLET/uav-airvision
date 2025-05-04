# src/image_processing/tracker.py

import cv2
import numpy as np
from config import VO_PARAMS

class FeatureTracker:
    def __init__(self):
        # FAST
        self.detector = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True
        )
        # Lucas–Kanade
        self.lk_params = {
            "winSize": VO_PARAMS["lk_win"],
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                VO_PARAMS["lk_criteria"][0],
                VO_PARAMS["lk_criteria"][1]
            )
        }

    def detect(self, img):
        kp = self.detector.detect(img)
        return np.array([p.pt for p in kp], dtype=np.float32)

    def track(self, img_ref, img_cur, pts_ref):
        from .utils import feature_tracking
        return feature_tracking(img_ref, img_cur, pts_ref, self.lk_params)
