# src/config.py

import cv2
import numpy as np
import yaml
import os

GRAVITY = np.array([0., 0., -9.81], dtype=np.float64)

# --- thresholds for FrameProcessor ---
MIN_INLIERS     = 5
FB_ERROR_THRESH = 1.0
MIN_TRACKED     = 50

# --- fallback intrinsics ---
FX, FY = 615.0, 615.0
CX, CY = 320.0, 240.0
CAMERA_MATRIX = np.array([
    [FX,  0, CX],
    [ 0, FY, CY],
    [ 0,  0,  1]
], dtype=np.float64)
DIST_COEFFS = np.zeros((4,1), dtype=np.float64)

# --- fallback extrinsic body→sensor ---
T_BS = np.eye(4, dtype=np.float64)

def load_euroc_calibration(yaml_path: str):
    """
    Reads EuRoC sensor.yaml and returns
      K     – 3×3 intrinsics,
      dist  – 4×1 radial‐tangential distortion,
      T_BS  – 4×4 body→sensor extrinsic.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"sensor.yaml not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    fu, fv, cu, cv = data['intrinsics']
    K = np.array([
        [fu, 0,  cu],
        [ 0, fv, cv],
        [ 0,  0,  1]
    ], dtype=np.float64)

    d = data['distortion_coefficients']
    dist = np.array(d[:4], dtype=np.float64).reshape(-1,1)

    tb = data['T_BS']['data']
    Tbs = np.array(tb, dtype=np.float64).reshape(4,4)

    return K, dist, Tbs

# try to load dataset calibration
USE_DATASET_CALIB = True
DATASET_PATH      = "datasets/MH_01_easy"
CAM_YAML          = os.path.join(DATASET_PATH, "mav0", "cam0", "sensor.yaml")

if USE_DATASET_CALIB:
    try:
        K_new, D_new, Tbs_new = load_euroc_calibration(CAM_YAML)
        CAMERA_MATRIX = K_new
        DIST_COEFFS   = D_new
        T_BS          = Tbs_new
        print(f"[config] Loaded EuRoC calibration from {CAM_YAML}")
    except Exception as e:
        print(f"[config] Warning: failed to load EuRoC calib: {e}")
        # keep fallbacks

# --------------- ORB / KLT / VIO parameters ----------------
N_FEATURES       = 15000
INIT_SCALE       = 1.2
WTA_K            = 2
SCORE_TYPE       = cv2.ORB_HARRIS_SCORE
PATCH_SIZE       = 31
FAST_THRESHOLD   = 17

E_RANSAC_THRESHOLD = 1.0

LK_WIN_SIZE      = (21,21)
LK_CRITERIA      = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

LAMBDA_REG       = 0.09
MAX_PTS_PER_CELL = 3
WINDOW_SIZE      = 14

VO_FPS           = 30.0
IMU_ACCEL_NOISE  = 0.05
VO_NOISE         = 0.03

SHI_QUALITY = 0.03
SHI_MIN_DIST = 7
SHI_BLOCK_SIZE = 7

RANSAC_THRESHOLD = 1.0
