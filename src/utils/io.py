from pathlib import Path
import numpy as np

def get_dataset_name(dataset_path: Path) -> str:
    return dataset_path.parent.name

def save_results_npz(
    pred_states, gt_states, pred_positions, gt_positions,
    dataset_name: str,
    base_dir: Path = Path("results") / "npz"
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    npz_path = base_dir / f"{dataset_name}_results.npz"

    pred_e = np.rad2deg(np.unwrap(pred_states[:, 3:6], axis=0))
    pred_v = pred_states[:, 6:9]
    gt_e   = np.rad2deg(np.unwrap(gt_states[:, :3], axis=0))
    gt_v   = gt_states[:, 3:6]

    np.savez(
        npz_path,
        pred_e=pred_e, gt_e=gt_e,
        pred_v=pred_v, gt_v=gt_v,
        pred_pos=pred_positions, gt_pos=gt_positions
    )
    return npz_path
