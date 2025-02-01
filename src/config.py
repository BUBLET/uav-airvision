import cv2
import numpy as np

VIDEO_PATH: str = "datasets/output.mp4"

FX: float = 615.0
FY: float = 615.0
CX: float = 320.0
CY: float = 240.0

CAMERA_MATRIX: np.ndarray = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float64)

DIST_COEFFS: np.ndarray = np.zeros((4, 1), dtype=np.float64)

# Параметры для ORB
N_FEATURES: int = 2000
KPTS_UNIFORM_SELECTION_GRID_SIZE: int = 16
MAX_PTS_PER_GRID: int = 2
INIT_SCALE: float = 1.2
WTA_K: int = 2
SCORE_TYPE: int = cv2.ORB_HARRIS_SCORE
PATCH_SIZE: int = 31
FAST_THRESHOLD: int = 10

E_RANSAC_THRESHOLD: float = 1.0

LK_WIN_SIZE = (21, 21)
LK_MAX_COUNT = 30
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


