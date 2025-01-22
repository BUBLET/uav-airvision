import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры камеры и пути
data_path = "./datasets/test_data"
image_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

focal_length = [615, 615]
principal_point = [320, 240]
camera_matrix = np.array([
    [focal_length[0], 0, principal_point[0]],
    [0, focal_length[1], principal_point[1]],
    [0, 0, 1]
], dtype=np.float32)
distortion_coeffs = np.zeros(5, dtype=np.float32)

def initialize(image: np.ndarray, camera_matrix, distortion_coeffs):
    if image is None:
        raise ValueError("Could not load image.")
    postI = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    postI = cv2.undistort(postI, camera_matrix, distortion_coeffs)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(postI, None)
    numPoints = 200
    if len(keypoints) > numPoints:
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        step_size = len(keypoints) // numPoints
        selected_keypoints = keypoints[::step_size][:numPoints]
        selected_indices = [keypoints.index(kp) for kp in selected_keypoints]
        descriptors = descriptors[selected_indices]
        keypoints = selected_keypoints
    vSet = {}
    # Сохраняем параметры камеры под ключом "pose"
    camera_pose = {
        "pose": {
            'R': np.eye(3),
            't': np.zeros((3,))
        },
        'keypoints': keypoints,
        'descriptors': descriptors
    }
    vSet[1] = camera_pose
    return vSet, keypoints, descriptors

def process_frame(image: np.ndarray, prev_points, prev_features, camera_matrix, distortion_coeffs, vSet, viewID):
    if image is None:
        raise ValueError("Could not load image.")
    postI = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    postI = cv2.undistort(postI, camera_matrix, distortion_coeffs)
    orb = cv2.ORB_create()
    curr_points, curr_features = orb.detectAndCompute(postI, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(prev_features, curr_features)
    matches = sorted(matches, key=lambda x: x.distance)
    prev_matched_points = np.float32([prev_points[m.queryIdx].pt for m in matches])
    curr_matched_points = np.float32([curr_points[m.trainIdx].pt for m in matches])
    essential_matrix, mask = cv2.findEssentialMat(
        prev_matched_points, curr_matched_points, camera_matrix,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential_matrix is None:
        raise RuntimeError("Essential Matrix couldn't be computed.")
    _, R, t, mask_inliers = cv2.recoverPose(essential_matrix, prev_matched_points, curr_matched_points, camera_matrix)
    inlier_matches = [m for i, m in enumerate(matches) if mask_inliers[i]]
    vSet[viewID] = {
        "pose": {"R": R, "t": t},
        "keypoints": curr_points,
        "descriptors": curr_features,
        "matches": inlier_matches
    }
    vSet[viewID - 1]["connections"] = inlier_matches
    return vSet, curr_points, curr_features, R, t

# --- Инициализация первого кадра ---
first_image_path = os.path.join(data_path, image_files[0])
first_image = cv2.imread(first_image_path)
vSet, prev_points, prev_features = initialize(first_image, camera_matrix, distortion_coeffs)

# Настройка 3D-визуализации
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-220, 50])
ax.set_ylim([-140, 20])
ax.set_zlim([-50, 300])
ax.view_init(elev=20, azim=-120)
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.grid(True)

trajectory_estimated = np.zeros((1, 3))  # Начальная точка траектории

# --- Обработка второго кадра отдельно ---
viewID = 2
second_image_path = os.path.join(data_path, image_files[1])
second_image = cv2.imread(second_image_path)
vSet, curr_points, curr_features, R, t = process_frame(
    second_image, prev_points, prev_features, camera_matrix, distortion_coeffs, vSet, viewID
)
# Аккумулируем глобальную позу для второго кадра
prev_pose = vSet[1]["pose"]
global_R = prev_pose["R"] @ R
global_t = prev_pose["t"] + prev_pose["R"].dot(t.flatten())
vSet[viewID]["pose"] = {"R": global_R, "t": global_t}
trajectory_estimated = np.vstack([trajectory_estimated, global_t])

# Обновление для следующего кадра
prev_points = curr_points
prev_features = curr_features

# --- Обработка кадров с 3 по 15 ---
for idx in range(2, min(15, len(image_files))):
    viewID = idx + 1  # Индекс для следующего кадра в нашей структуре
    image_path = os.path.join(data_path, image_files[idx])
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        break

    vSet, curr_points, curr_features, R, t = process_frame(
        image, prev_points, prev_features, camera_matrix, distortion_coeffs, vSet, viewID
    )

    # Аккумуляция глобальной позы
    prev_pose = vSet[viewID - 1]["pose"]
    global_R = prev_pose["R"] @ R
    global_t = prev_pose["t"] + prev_pose["R"].dot(t.flatten())
    vSet[viewID]["pose"] = {"R": global_R, "t": global_t}

    # Обновление траектории
    trajectory_estimated = np.vstack([trajectory_estimated, global_t])

    # Обновление графика
    ax.plot(trajectory_estimated[:, 0], trajectory_estimated[:, 1], trajectory_estimated[:, 2], 'g-')
    plt.pause(0.1)

    # Подготовка к следующей итерации
    prev_points = curr_points
    prev_features = curr_features

# Финальная визуализация
ax.legend(['Estimated Trajectory'])
plt.title('Camera Trajectory (Frames 1 to 15)')
plt.show()
