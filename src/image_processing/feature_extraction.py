import cv2
import numpy as np
from typing import Tuple, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_uniform_keypoints_by_grid(
        keypoints: List[cv2.KeyPoint],
        image_rows: int,
        image_cols: int,
        grid_size: int,
        max_pts_per_cell: int
) -> List[cv2.KeyPoint]:
    rows = image_rows // grid_size
    cols = image_cols // grid_size
    if rows == 0 or cols == 0:
        return keypoints
    grid = np.zeros((rows, cols), dtype=np.int32)
    selected_keypoints = []
    for kpt in keypoints:
        row_idx = int(kpt.pt[1]) // grid_size
        col_idx = int(kpt.pt[0]) // grid_size
        if row_idx < 0 or row_idx >= rows or col_idx < 0 or col_idx >= cols:
            continue
        if grid[row_idx, col_idx] < max_pts_per_cell:
            selected_keypoints.append(kpt)
            grid[row_idx, col_idx] += 1
    return selected_keypoints

class FeatureExtractor:
    def __init__(
            self,
            extractor: Any = None,
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

        if extractor is None:
            self.extractor = cv2.ORB_create(
                nfeatures=nfeatures,
                scaleFactor=scaleFactor,
                nlevels=nlevels,
                edgeThreshold=edgeThreshold,
                firstLevel=firstLevel,
                WTA_K=WTA_K,
                scoreType=scoreType,
                patchSize=patchSize,
                fastThreshold=fastThreshold
            )
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
        keypoints = self.extractor.detect(image_gray, None)
        logger.info(f"Обнаружено {len(keypoints)} точек до распределения")
    
        keypoints, descriptors = self.extractor.compute(image_gray, keypoints)
        if descriptors is None:
            logger.warning("Нет дескрипторов после фильтрации")
        return keypoints, descriptors

