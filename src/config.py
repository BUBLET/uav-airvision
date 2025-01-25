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

LOWE_RATIO: float = 0.9
KNN_K: int = 2
N_FEATURES: int = 9000
N_LEVELS: int = 8
EDGE_THRESHOLD: int = 8

KPTS_UNIFORM_SELECTION_GRID_SIZE: int = 48
MAX_PTS_PER_GRID: int = 2
INIT_SCALE: float = 1.0
WTA_K: int = 3
SCORE_TYPE: int = cv2.ORB_HARRIS_SCORE
PATCH_SIZE: int = 31
FAST_THRESHOLD: int = 30

E_RANSAC_THRESHOLD: float = 0.6
H_RANSAC_THRESHOLD: float = 0.1
DISTANCE_THRESHOLD: float = 77.0
MAP_CLEAN_MAX_DISTANCE: float = 195.0
REPROJECTION_THRESHOLD: float = 4.72
RATIO_THRESH: float = 0.45
CKD_RADIUS: float = 5.0

TRANSLATION_THRESHOLD: float = 1.4
ROTATION_THRESHOLD_DEG: float = 2.75
ROTATION_THRESHOLD: float = np.deg2rad(ROTATION_THRESHOLD_DEG)
TRIANGULATION_THRESHOLD_DEG: float = 2.0
TRIANGULATION_THRESHOLD: float = np.deg2rad(TRIANGULATION_THRESHOLD_DEG)
BUNDLE_ADJUSTMENT_FRAMES: int = 12
FORCE_KEYFRAME_INTERVAL: int = 1
HOMOGRAPHY_INLIER_RATIO: float = 0.86

BA_MAX_NFEV: int = 100
BA_FTOL: float = 0.1
BA_XTOL: float = 0.008
BA_GTOL: float = 0.085

LOST_THRESHOLD: int = 10
EPIPOLAR_THRESHOLD: float = 0.082
ASSUMED_MEAN_DEPTH_DURING_INIT: float = 0.8

MAX_TOTAL_POINTS: int = 90000
MIN_OBSERVATIONS: int = 6
MAX_REPROJ_ERROR: float = 90.0


ORB_INTERVAL: int = 1
KEYFRAME_BA_INTERVAL: int = 7
MAX_POINTS: int = 10000
