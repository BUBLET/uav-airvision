import cv2
import numpy as np
from typing import Tuple, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(
            self,
            extractor: Any = None,
            grid_size: int = 16,
            max_pts_per_cell: int = 2,
            nfeatures: int = 8000,
            scaleFactor: float = 1.2,
            nlevels: int = 8,
            edgeThreshold: int = 15,
            firstLevel: int = 0,
            WTA_K: int = 2,
            scoreType: int = cv2.ORB_HARRIS_SCORE,
            patchSize: int = 31,
            fastThreshold: int = 10
    ):
        self.grid_size = grid_size
        self.max_pts_per_cell = max_pts_per_cell
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
            self.extractor = cv2.ORB_create(
                nfeatures=self.nfeatures,
                scaleFactor=self.scaleFactor,
                nlevels=self.nlevels,
                edgeThreshold=self.edgeThreshold,
                firstLevel=self.firstLevel,
                WTA_K=self.WTA_K,
                scoreType=self.scoreType,
                patchSize=self.patchSize,
                fastThreshold=self.fastThreshold
            )
            logger.info("FeatureExtractor инициализирован с ORB.")
        else:
            self.extractor = extractor
            logger.info("FeatureExtractor инициализирован с пользовательским детектором.")

    def adapt_threshold(self, image_gray: np.ndarray):
        avg_intensity = float(np.mean(image_gray))
        desired_intensity = 128.0
        factor = desired_intensity / (avg_intensity + 1e-5)
        new_threshold = int(self.fastThreshold * factor)
        new_threshold = max(5, min(new_threshold, 50))
        if new_threshold != self.fastThreshold:
            logger.info(f"Изменение fastThreshold: {self.fastThreshold} -> {new_threshold} (средняя яркость = {avg_intensity:.2f})")
            self.fastThreshold = new_threshold
            self.extractor = cv2.ORB_create(
                nfeatures=self.nfeatures,
                scaleFactor=self.scaleFactor,
                nlevels=self.nlevels,
                edgeThreshold=self.edgeThreshold,
                firstLevel=self.firstLevel,
                WTA_K=self.WTA_K,
                scoreType=self.scoreType,
                patchSize=self.patchSize,
                fastThreshold=self.fastThreshold
            )

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


        self.adapt_threshold(image_gray)

        keypoints = self.extractor.detect(image_gray, None)
        logger.info(f"Обнаружено {len(keypoints)} точек до распределения")
    
        keypoints, descriptors = self.extractor.compute(image_gray, keypoints)
        if descriptors is None:
            logger.warning("Нет дескрипторов после фильтрации")
        return keypoints, descriptors
