import numpy as np
from typing import List, Tuple
import itertools

class MapPoint:
    _id_generator = itertools.count()

    def __init__(self, 
                 coordinates: np.ndarray
    ):
        
        self.id = next(MapPoint._id_generator)
        self.coordinates = coordinates  
        self.descriptors: List[np.ndarray] = []  
        self.observations: List[Tuple[int, int]] = []  
        self.matched_times: int = 0

    def add_observation(self, frame_idx: int, keypoint_idx: int):
        self.observations.append((frame_idx, keypoint_idx))

    def is_frequently_matched(self, threshold: int = 3):
        return self.matched_times >= threshold

    def __repr__(self):
        return f"MapPoint(coordinates={self.coordinates}, descriptors={len(self.descriptors)}, observations={len(self.observations)})"
