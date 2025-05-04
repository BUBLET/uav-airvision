import cv2
from pathlib import Path

from utils import get_dataset_name, save_results_npz, run_pipeline
from gen_results import generate_results
from config import DATASET_PATH

def main():
    pred_states, gt_states, pred_positions, gt_positions = run_pipeline()

    dataset_name = get_dataset_name(Path(DATASET_PATH))
    npz_path = save_results_npz(
        pred_states, gt_states,
        pred_positions, gt_positions,
        dataset_name
    )

    generate_results(str(npz_path), base_output_dir="results")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
