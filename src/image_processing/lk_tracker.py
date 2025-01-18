import cv2
import numpy as np
import config

class LKTracker:
    def __init__(self):
        self.lk_params = dict(
            winSize=config.LK_WIN_SIZE,
            maxLevel=config.LK_MAX_LEVEL,
            criteria=config.LK_CRITERIA
        )

    def track(self, prev_frame, current_frame, prev_points):
        """
        prev_points: numpy array of shape (N, 2) — координаты x,y
        Возвращает curr_points и статус
        """
        curr_points, st, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, current_frame, prev_points, None, **self.lk_params
        )

        # Фильтруем только те точки, которые успешно оттракались
        status_mask = st.reshape(-1).astype(bool)
        curr_points_good = curr_points[status_mask]
        prev_points_good = prev_points[status_mask]

        return prev_points_good, curr_points_good
