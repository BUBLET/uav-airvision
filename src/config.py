# config.py

from pathlib import Path

# Пути
DATASET_PATH = Path("datasets/MH_01_easy/mav0/")
OUTPUT_TRAJ = {
    "xy": Path("map_xy.png"),
    "xz": Path("map_xz.png"),
    "yz": Path("map_yz.png"),
}

# Камера
CAM_PARAMS = {
    "width": 752,
    "height": 480,
    "fx": 458.654,
    "fy": 457.296,
    "cx": 367.215,
    "cy": 248.375,
    "dist": {"k1": -0.28340811, "k2": 0.07395907,
             "p1": 0.00019359, "p2": 1.76187114e-05, "k3": 0.0}
}

# Visual Odometry
VO_PARAMS = {
    "min_features": 2500,
    "lk_win": (5, 5),
    "lk_criteria": (30, 0.01),
    "ess_threshold": 0.7,
    "ess_prob": 0.999,
    "clamp_deg": 5.0,
    "T_BS": [
        [ 0.01517066, -0.99983694,  0.00979558, -0.01638528],
        [ 0.99965712,  0.01537559,  0.02119505, -0.06812726],
        [-0.02134221,  0.00947067,  0.99972737,  0.00395795],
        [ 0.0,          0.0,         0.0,         1.0        ]
    ]
}

# Preprocessing
IMU_TIMESHIFT_S = 5.63799926987e-05
