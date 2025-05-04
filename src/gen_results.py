import os
import numpy as np
import matplotlib.pyplot as plt

def generate_results(
    npz_file: str = "vo_results.npz",
    base_output_dir: str = "results"
):

    dataset_name = os.path.splitext(os.path.basename(npz_file))[0]
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_file)
    pred_e = data["pred_e"]
    gt_e   = data["gt_e"]
    pred_v = data["pred_v"]
    gt_v   = data["gt_v"]
    pred_p = data["pred_pos"]
    gt_p   = data["gt_pos"]

    N = pred_e.shape[0]
    xs = np.arange(N)
    titles_e = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    titles_v = ['Vx (m/s)',    'Vy (m/s)',    'Vz (m/s)']

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for i in range(3):
        ax = axs[0, i]
        ax.plot(xs, pred_e[:, i], label='VO')
        ax.plot(xs, gt_e[:, i],   label='GT')
        ax.set_title(titles_e[i])
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Angle, deg')
        ax.grid(True)
        ax.legend()

        ax = axs[1, i]
        ax.plot(xs, pred_v[:, i], label='VO')
        ax.plot(xs, gt_v[:, i],   label='GT')
        ax.set_title(titles_v[i])
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Speed, m/s')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    out1 = os.path.join(output_dir, "euler_velocity_comparison.png")
    plt.savefig(out1, dpi=300)
    plt.close(fig)

    diff_gt = np.linalg.norm(gt_p[1:]  - gt_p[:-1],  axis=1)
    diff_vo = np.linalg.norm(pred_p[1:] - pred_p[:-1], axis=1)
    cum_gt  = np.concatenate([[0], np.cumsum(diff_gt)])
    cum_vo  = np.concatenate([[0], np.cumsum(diff_vo)])
    eps = 1e-6
    error_pct = np.abs(cum_vo - cum_gt) / (cum_gt + eps) * 100

    fig2 = plt.figure(figsize=(10,4))
    plt.plot(cum_gt,   label='GT path length (m)')
    plt.plot(cum_vo,   label='VO path length (m)')
    plt.plot(error_pct, label='Relative error (%)')
    plt.xlabel('Frame index')
    plt.ylabel('Percent')
    plt.title('Cumulative Path and Relative Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "path_error.png")
    plt.savefig(out2, dpi=300)
    plt.close(fig2)


if __name__ == "__main__":
    generate_results()
