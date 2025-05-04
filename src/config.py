from pathlib import Path
import yaml

DATASET_PATH = Path("datasets/MH_01_easy/mav0/")
OUTPUT_TRAJ = {
    "xy": Path("map_xy.png"),
    "xz": Path("map_xz.png"),
    "yz": Path("map_yz.png"),
}

SENSOR_YAML = DATASET_PATH / "cam0" / "sensor.yaml"

def load_camera_params(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    tb = data["T_BS"]["data"]
    T_BS = [tb[i*4:(i+1)*4] for i in range(4)]

    fu, fv, cu, cv = data["intrinsics"]
    dist = data.get("distortion_coefficients", [])
    if len(dist) == 4:
        k1, k2, p1, p2 = dist
        k3 = 0.0
    elif len(dist) == 5:
        k1, k2, p1, p2, k3 = dist
    else:
        k1 = k2 = p1 = p2 = k3 = 0.0

    width, height = data["resolution"]

    return {
        "T_BS": T_BS,
        "width": width,
        "height": height,
        "fx": fu,
        "fy": fv,
        "cx": cu,
        "cy": cv,
        "dist": {"k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3},
    }

_cam = load_camera_params(SENSOR_YAML)

CAM_PARAMS = {
    "width":  _cam["width"],
    "height": _cam["height"],
    "fx":      _cam["fx"],
    "fy":      _cam["fy"],
    "cx":      _cam["cx"],
    "cy":      _cam["cy"],
    "dist":    _cam["dist"],
}

VO_PARAMS = {
    "min_features": 3000,
    "lk_win": (5, 5),
    "lk_criteria": (40, 0.01),
    "ess_threshold": 0.7,
    "ess_prob": 0.999,
    "clamp_deg": 5.0,
    "T_BS": _cam["T_BS"],
}

IMU_TIMESHIFT_S = 5.63799926987e-05
