import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_ATE(gt, vo):
    """
    Compute per-frame Absolute Trajectory Error (ATE) by aligning VO to GT
    using Horn's closed-form solution.
    """
    # 1) центры
    gt_mean = gt.mean(axis=0)
    vo_mean = vo.mean(axis=0)

    # 2) центрированные траектории
    gt_centered = gt - gt_mean
    vo_centered = vo - vo_mean

    # 3) корреляционная матрица
    H = vo_centered.T @ gt_centered

    # 4) SVD и оптимальное вращение
    U, _, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # 5) оптимальный сдвиг
    t_opt = gt_mean - R_opt @ vo_mean

    # 6) выравнивание и вычисление ошибок
    vo_aligned = (R_opt @ vo.T).T + t_opt
    errors = np.linalg.norm(gt - vo_aligned, axis=1)
    return errors

def compute_drift_per_meter(ate_rmse, gt):
    """
    Compute overall drift per meter: ATE_RMSE / total_path_length.
    """
    total_length = np.sum(np.linalg.norm(gt[1:] - gt[:-1], axis=1))
    return ate_rmse / (total_length + 1e-9)

def generate_results(
    npz_file: str = "vo_results.npz",
    base_output_dir: str = "results"
):
    # Папка для результатов
    dataset_name = os.path.splitext(os.path.basename(npz_file))[0]
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка данных
    data   = np.load(npz_file)
    pred_e = data["pred_e"]
    gt_e   = data["gt_e"]
    pred_v = data["pred_v"]
    gt_v   = data["gt_v"]
    pred_p = data["pred_pos"]
    gt_p   = data["gt_pos"]

    # ── 1) Сравнение углов и скоростей ──────────────────────────────────────
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

    # ── 2) Cumulative path & relative error ─────────────────────────────────
    diff_gt = np.linalg.norm(gt_p[1:] - gt_p[:-1], axis=1)
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

    # ── 3) Absolute Trajectory Error: метры и проценты ─────────────────────────
    errors = compute_ATE(gt_p, pred_p)                  # [m]
    diff_gt = np.linalg.norm(gt_p[1:] - gt_p[:-1], axis=1)
    cum_gt  = np.concatenate([[0], np.cumsum(diff_gt)])  # [m]

    # ATE в процентах
    with np.errstate(divide='ignore', invalid='ignore'):
        errors_pct = np.where(cum_gt > 0, errors / cum_gt * 100.0, 0.0)

    # Статистики
    ate_mean     = errors.mean()
    ate_rmse     = np.sqrt((errors**2).mean())
    ate_pct_mean = errors_pct.mean()
    ate_pct_rmse = np.sqrt((errors_pct**2).mean())

    # Рисуем оба графика в одном файле
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Сабплот 1: абсолютная ошибка
    axs[0].plot(errors,       label=f'ATE (m): mean={ate_mean:.3f}, rmse={ate_rmse:.3f}')
    axs[0].set_ylabel('ATE error (m)')
    axs[0].set_title('Absolute Trajectory Error per Frame')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Сабплот 2: ошибка в процентах
    axs[1].plot(errors_pct,   label=f'ATE (%): mean={ate_pct_mean:.2f}%, rmse={ate_pct_rmse:.2f}%')
    axs[1].set_xlabel('Frame index')
    axs[1].set_ylabel('ATE error (%)')
    axs[1].set_title('Absolute Trajectory Error per Frame (percent)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ate_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


    # ── 4) Drift per Meter (DpM) ─────────────────────────────────────────────
    dpm = compute_drift_per_meter(ate_rmse, gt_p)
    cumulative_length = np.concatenate([[0], np.cumsum(diff_gt)])
    cumulative_drift = np.cumsum(errors) / (cumulative_length + 1e-9)

    plt.figure(figsize=(8,4))
    plt.plot(cumulative_length, cumulative_drift)
    plt.xlabel("GT cumulative path length (m)")
    plt.ylabel("Cumulative drift per meter")
    plt.title("Drift per Meter over Path")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dpm_over_path.png"))
    plt.close()

    # ── 5) Сохраняем итоговую таблицу метрик ─────────────────────────────────
    df = pd.DataFrame({
        "Metric": [
            "ATE Mean (m)", "ATE RMSE (m)",
            "ATE Mean (%)", "ATE RMSE (%)",
            "Drift per meter (m/m)"
        ],
        "Value": [
            errors.mean(), np.sqrt((errors**2).mean()),
            ate_pct_mean, ate_pct_rmse,
            dpm
        ]
    })
    df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

if __name__ == "__main__":
    generate_results()
