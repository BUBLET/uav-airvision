import cv2
import numpy as np

VIDEO_PATH = "D:/Мисис/Диплом/AirVision/datasets/output.mp4"

FX = 615.0
FY = 615.0
CX = 320.0
CY = 240.0

CAMERA_MATRIX = np.array([[FX, 0, CX],
                          [0, FY, CY],
                          [0, 0, 1]
                          ], dtype=np.float64)

DIST_COEFFS = np.zeros((4,1), dtype=np.float64)

LOWE_RATIO = 0.8
KNN_K = 2

TRANSLATION_THRESHOLD = 0.005
ROTATION_THRESHOLD_DEG = 0.5
ROTATION_THRESHOLD = np.deg2rad(ROTATION_THRESHOLD_DEG)

TRIANGULATION_THRESHOLD_DEG = 1.0
TRIANGULATION_THRESHOLD = np.deg2rad(TRIANGULATION_THRESHOLD_DEG)

BUNDLE_ADJUSTMENT_FRAMES = 5
ORB_INTERVAL = 1
FORCE_KEYFRAME_INTERVAL = 1
KEYFRAME_BA_INTERVAL = 5

MAX_REPROJ_ERROR = 100.0
MIN_OBSERVATIONS = 1
MAX_POINTS = 10000

MAP_CLEAN_MAX_DISTANCE = 100.0

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01
)

BA_MAX_NFEV = 100
BA_FTOL = 1e-2

E_RANSAC_THRESHOLD = 0.5
H_RANSAC_THRESHOLD = 2.0
HOMOGRAPHY_INLIER_RATIO = 0.45

# Параметры ре-инициализации
LOST_THRESHOLD = 5

INIT_SCALE = 1.0
ASSUMED_MEAN_DEPTH_DURING_INIT = 0.5

KPTS_UNIFORM_SELECTION_GRID_SIZE = 16
MAX_PTS_PER_GRID = 8
MAX_TOTAL_POINTS = 4000

