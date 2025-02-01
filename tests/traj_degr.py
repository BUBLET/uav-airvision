import os
import numpy as np
import matplotlib.pyplot as plt


def parse_trajectory_file(file_path):
    """
    Считывает файл траектории, где каждая строка содержит 12 чисел:
    tx, ty, tz, R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1], R[0,2], R[1,2], R[2,2].
    Возвращает:
    positions (array, shape=(N,3)),
    rotations (array, shape=(N,3,3)).
    """
    positions = []
    rotations = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) != 12:
                continue

            vals = list(map(float, vals))
            tx, ty, tz = vals[0], vals[1], vals[2]
            r00, r10, r20 = vals[3], vals[4], vals[5]
            r01, r11, r21 = vals[6], vals[7], vals[8]
            r02, r12, r22 = vals[9], vals[10], vals[11]

            R = np.array([
                [r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]
            ], dtype=np.float64)
            t = np.array([tx, ty, tz], dtype=np.float64)

            positions.append(t)
            rotations.append(R)

    return np.array(positions), np.array(rotations)



def main():
    # Пути к файлам
    gt_file = os.path.join("results", "cam_traj_truth.txt")
    est_file = os.path.join("trajectory.txt")

    # Считываем
    gt_positions, _ = parse_trajectory_file(gt_file)
    est_positions, _ = parse_trajectory_file(est_file)

    # Сопоставим длины
    n = min(len(gt_positions), len(est_positions))
    frames = np.arange(n)

    gt_positions = gt_positions[:n]
    est_positions = est_positions[:n]

 

    fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True)
    fig.suptitle("Top row: GT  |  Middle row: Estimated ")

    # X
    axes[0, 0].plot(frames, gt_positions[:, 0], color='blue')
    axes[0, 0].set_title("GT X")
    axes[0, 0].grid(True)

    # Y
    axes[0, 1].plot(frames, gt_positions[:, 1], color='blue')
    axes[0, 1].set_title("GT Y")
    axes[0, 1].grid(True)

    # Z
    axes[0, 2].plot(frames, gt_positions[:, 2], color='blue')
    axes[0, 2].set_title("GT Z")
    axes[0, 2].grid(True)

    # --- Estimated (вторая строка, row=1) ---
    # X
    axes[1, 0].plot(frames, est_positions[:, 0], color='red')
    axes[1, 0].set_title("Est X")
    axes[1, 0].grid(True)

    # Y
    axes[1, 1].plot(frames, est_positions[:, 1], color='red')
    axes[1, 1].set_title("Est Y")
    axes[1, 1].grid(True)

    # Z
    axes[1, 2].plot(frames, est_positions[:, 2], color='red')
    axes[1, 2].set_title("Est Z")
    axes[1, 2].grid(True)


    for col in range(3):
        axes[1, col].set_xlabel("Frame Index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
