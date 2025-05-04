import cv2
import numpy as np
from typing import Tuple, List, Optional, Any
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(
            self,
            extractor: Any = None,
            nfeatures = config.N_FEATURES,
            scaleFactor: float = 1.2,
            nlevels: int = 8,
            edgeThreshold: int = 15,
            firstLevel: int = 0,
            WTA_K: int = 2,
            scoreType: int = cv2.ORB_HARRIS_SCORE,
            patchSize: int = 31,
            fastThreshold = config.FAST_THRESHOLD
    ):
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.edgeThreshold = edgeThreshold
        self.firstLevel = firstLevel
        self.WTA_K = WTA_K
        self.scoreType = scoreType
        self.patchSize = patchSize
        self.fastThreshold = fastThreshold  # начальное значение порога

        if extractor is None:
            self.extractor = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            logger.info("FeatureExtractor инициализирован с ORB.")
        else:
            self.extractor = extractor
            logger.info("FeatureExtractor инициализирован с пользовательским детектором.")

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Изображение пустое.")
        if image.size == 0:
            raise ValueError("Изображение пустое.")
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Изображение преобразовано в градации серого.")
        elif len(image.shape) == 2:
            image_gray = image
        else:
            raise ValueError("Неподдерживаемый формат изображения.")
        
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_preprocessed = image_gray

        keypoints = self.extractor.detect(image_preprocessed, None)
        logger.info(f"Обнаружено {len(keypoints)} точек.")

        keypoints, descriptors = self.extractor.compute(image_preprocessed, keypoints)
        if descriptors is None:
            logger.warning("Нет дескрипторов после фильтрации")
        return keypoints, descriptors
