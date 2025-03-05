import cv2
import numpy as np

FRAMES_FOLDER = ""
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
N_FEATURES = 13000
INIT_SCALE: float = 1.2
WTA_K: int = 2
SCORE_TYPE: int = cv2.ORB_HARRIS_SCORE
PATCH_SIZE: int = 31
FAST_THRESHOLD = 17

E_RANSAC_THRESHOLD = 0.7

LK_WIN_SIZE = (21, 21)
LK_MAX_COUNT = 20
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

LAMBDA_REG = 0.09
MAX_PTS_PER_CELL = 2
WINDOW_SIZE = 14
