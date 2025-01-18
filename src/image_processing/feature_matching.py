import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import config

class FeatureMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.logger = logging.getLogger(__name__)

    def match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ):

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Дескрипторы не могут быть None.")

        if len(descriptors1) == 0 or len(descriptors2) is None:
            self.logger.warning("descriptors are None")
            return []

        # Выполняем KNN-сопоставление
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=config.KNN_K)
        if len(matches) == 0:
            self.logger.warning("no matches")
            return []
        
        # Применяем тест соотношения Лоу
        good_matches = []
        for m, n in matches:
            if m.distance < config.LOWE_RATIO * n.distance:
                good_matches.append(m)

        self.logger.info(f"Найдено {len(good_matches)} хороших соответствий.")

        return good_matches
