import numpy as np
import cv2

def feature_tracking(img_ref, img_cur, pts_ref, lk_params):
    kp2, st, _ = cv2.calcOpticalFlowPyrLK(
        img_ref, img_cur, pts_ref, None, **lk_params
    )
    st = st.ravel() == 1
    return pts_ref[st], kp2[st]

def rotation_matrix_to_euler(R):
    sy = np.hypot(R[0,0], R[1,0])
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0
    return np.array([roll, pitch, yaw], dtype=float)

def euler_to_rotation_matrix(e):
    roll, pitch, yaw = e
    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,            0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1,           0,            0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    return Rz @ Ry @ Rx

def clamp_euler(old, new, max_deg):
    max_rad = np.deg2rad(max_deg)
    diff = (new - old + np.pi) % (2*np.pi) - np.pi
    clamped = old.copy()
    for i in range(3):
        if abs(diff[i]) <= max_rad:
            clamped[i] = new[i]
        else:
            clamped[i] = old[i] + np.sign(diff[i]) * max_rad
    return clamped
