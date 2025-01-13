import cv2
import numpy as np

class LKTracker:
    def __init__(self,
                 win_size=(21, 21),
                 max_level=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)):
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=criteria
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
