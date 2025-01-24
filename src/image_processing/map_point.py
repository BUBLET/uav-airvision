import numpy as np
from typing import List, Tuple

class MapPoint:
    def __init__(self, 
                 id_: int,
                 coordinates: np.ndarray
    ):
        
        self.id = id_
        self.coordinates = coordinates  # 3D координаты точки
        self.descriptors: List[np.ndarray] = []  # Список дескрипторов этой точки
        self.observations: List[Tuple[int, int]] = []  # Список наблюдений в кадрах
        self.matched_times: int = 0

    def add_observation(self, frame_idx: int, keypoint_idx: int):
        self.observations.append((frame_idx, keypoint_idx))

    def is_frequently_matched(self, threshold: int = 3):
        return self.matched_times >= threshold

    def __repr__(self):
        return f"MapPoint(coordinates={self.coordinates}, descriptors={len(self.descriptors)}, observations={len(self.observations)})"
