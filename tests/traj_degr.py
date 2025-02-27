import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def parse_trajectory_file(file_path):
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                positions.append(vals[:3])
    return np.array(positions)

def main():
    gt_positions = parse_trajectory_file("results/cam_traj_truth.txt")
    est_positions = parse_trajectory_file("results/estimated_traj.txt")
    
    n = min(len(gt_positions), len(est_positions))
    gt_positions, est_positions = gt_positions[:n], est_positions[:n]
    frames = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True)
    for i, label in enumerate(["X", "Y", "Z"]):
        axes[0, i].plot(frames, gt_positions[:, i], color='blue')
        axes[1, i].plot(frames, est_positions[:, i], color='red')
        axes[0, i].set_title(f"GT {label}")
        axes[1, i].set_title(f"Est {label}")
        axes[0, i].grid(True)
        axes[1, i].grid(True)
        axes[1, i].set_xlabel("Frame Index")

    plt.tight_layout()
    plt.show()

    fig3d = plt.figure(figsize=(10, 8))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label="Ground Truth", color='blue')
    ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], label="Estimated", color='red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
