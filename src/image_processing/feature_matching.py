import cv2
import numpy as np
from typing import List, Tuple, Optional, Any
import logging

class FeatureMatcher:
    def __init__(self,
                 knn_k: int = 2,
                 lowe_ratio: float = 0.75,
                 norm_type: int = cv2.NORM_HAMMING,
                 matcher: Any = None,
                 logger: Optional[logging.Logger] = None
    ):
        self.knn_k = knn_k
        self.lowe_ratio = lowe_ratio
        self.norm_type = norm_type

        if matcher is not None:
            self.matcher = matcher
            self.logger = logging.getLogger(__name__)
            self.logger.info("FeatureMatcher инициализирован с переданным matcher")
        
        else:
            self.matcher = cv2.BFMatcher(self.norm_type)
            self.logger = logger or logging.getLogger(__name__)
            self.logger.info(f"FeatureMatch инициализирован с matcher по умолчанию")


    def match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Дескрипторы не могут быть None.")

        if len(descriptors1) == 0 or len(descriptors2) == 0:
            self.logger.warning("descriptors are empty")
            return []

        # Выполняем KNN-сопоставление
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=self.knn_k)
        if len(matches) == 0:
            self.logger.warning("no matches")
            return []
        
        # Применяем тест соотношения Лоу
        good_matches = []
        for m, n in matches:
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)

        self.logger.info(f"Найдено {len(good_matches)} хороших соответствий")
        self.logger.debug(f"Общее число пар для Лоу-теста: {len(matches)}, осталось good: {len(good_matches)}")  
        return good_matches
