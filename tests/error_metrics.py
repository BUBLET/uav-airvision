import numpy as np

# Задаём пути к файлам траекторий
GROUND_TRUTH_PATH = "./results/cam_traj_truth.txt"
ESTIMATED_PATH = "./results/estimated_traj.txt"

def load_trajectory(file_path: str) -> np.ndarray:

    data = np.loadtxt(file_path)
    return data[:, :3]

def compute_path_length(traj: np.ndarray) -> float:

    if traj.shape[0] < 2:
        return 0.0
    diffs = np.diff(traj, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

def procrustes_alignment(X: np.ndarray, Y: np.ndarray):

    # Центрирование
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY

    # Нормы
    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)
    if normX == 0 or normY == 0:
        raise ValueError("Невозможно нормировать траекторию с нулевой длиной.")
    X0 /= normX
    Y0 /= normY

    # Определяем матрицу поворота через SVD
    U, _, Vt = np.linalg.svd(Y0.T @ X0)
    R = U @ Vt

    # Оптимальный масштаб
    s = np.sum(X0 * (Y0 @ R)) / np.sum(Y0**2)

    # Применяем трансформацию: сначала масштабируем и поворачиваем, затем возвращаем центр X
    Y_aligned = s * (Y0 @ R) * normX + muX

    return Y_aligned, s, R, muX, muY

def compute_normalized_trajectory_error() -> float:

    gt_traj = load_trajectory(GROUND_TRUTH_PATH)
    est_traj = load_trajectory(ESTIMATED_PATH)

    # Приводим траектории к одинаковому количеству точек
    n_points = min(gt_traj.shape[0], est_traj.shape[0])
    gt_traj = gt_traj[:n_points]
    est_traj = est_traj[:n_points]

    # Выравниваем оценённую траекторию по истинной
    est_aligned, s, R, muX, muY = procrustes_alignment(gt_traj, est_traj)

    # Вычисляем среднюю ошибку по точкам
    errors = np.linalg.norm(gt_traj - est_aligned, axis=1)
    mean_error = np.mean(errors)
    L_gt = compute_path_length(gt_traj)

    if L_gt == 0:
        raise ValueError("Суммарная длина истинной траектории равна нулю.")

    # Ошибка в процентах относительно суммарной длины истинной траектории
    error_percentage = (mean_error / L_gt) * 100
    return error_percentage

if __name__ == "__main__":
    error = compute_normalized_trajectory_error()
    print("Ошибка траектории после нормализации: {:.2f}%".format(error))
