import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
import config

# Настройка логирования
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
    count_total = 0

    for kpt in  keypoints:
        row_idx = int(kpt.pt[1]) // grid_size
        col_idx = int(kpt.pt[0]) // grid_size

        if row_idx < 0 or row_idx >= rows or col_idx < 0 or col_idx >= cols:
            continue

        if grid[row_idx, col_idx] < max_pts_per_cell:
            selected_keypoints.append(kpt)
            grid[row_idx, col_idx] += 1
            count_total += 1

    return selected_keypoints

class FeatureExtractor:
    def __init__(self):
        self.extractor = cv2.ORB_create(
            nfeatures=8000, 
            scaleFactor=1.2, 
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        logger.info("FeatureExtractor инициализирован с параметрами ORB.")

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Извлекает ключевые точки и дескрипторы из изображения с использованием ORB.

        Параметры:
        - image (numpy.ndarray): Входное изображение в формате BGR или градаций серого.

        Возвращает:
        - keypoints (list of cv2.KeyPoint): Список найденных ключевых точек.
        - descriptors (numpy.ndarray или None): Массив дескрипторов.
        """

        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Изображение пустое.")

        if image.size == 0:
            raise ValueError("Изображение пустое.")

        # Преобразуем изображение в градации серого, если оно цветное
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Изображение преобразовано в градации серого.")
        elif len(image.shape) == 2:
            image_gray = image
        else:
            raise ValueError("Неподдерживаемый формат изображения.")

        # Извлекаем ключевые точки 
        keypoints = self.extractor.detect(image_gray, None)
        logger.info(f"Обнаружено {len(keypoints)} точек до распределения")

        filtered_keypoints = select_uniform_keypoints_by_grid(
            keypoints,
            image_gray.shape[0],
            image_gray.shape[1],
            config.KPTS_UNIFORM_SELECTION_GRID_SIZE,
            config.MAX_PTS_PER_GRID
        )
        logger.info(f"После распределения осталось {len(filtered_keypoints)} точек")

        descriptors = None
        if filtered_keypoints:
            filtered_keypoints, descriptors = self.extractor.compute(image_gray, filtered_keypoints)
            if descriptors is None:
                logger.warning("Нет дескрипторов после фильтрации")
        else:
            logger.warning("Нет точек для дескрипторов после фильтрации")
        
        return filtered_keypoints, descriptors

