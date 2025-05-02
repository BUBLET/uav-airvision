#!/usr/bin/env python3
import argparse
import os

from image_processing.visualizer import TrajectoryVisualizer

def main():
    parser = argparse.ArgumentParser(description="Визуализация траектории")
    parser.add_argument(
        "--gt-csv",
        required=True,
        help="Путь к ground-truth CSV (например: datasets/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv)"
    )
    parser.add_argument(
        "--est-txt",
        required=True,
        help="Путь к файлу с оценённой траекторией (например: results/estimated_traj.txt)"
    )
    args = parser.parse_args()

    # Проверяем, что файлы есть
    if not os.path.isfile(args.gt_csv):
        raise FileNotFoundError(f"GT-файл не найден: {args.gt_csv}")
    if not os.path.isfile(args.est_txt):
        raise FileNotFoundError(f"Est-файл не найден: {args.est_txt}")

    viz = TrajectoryVisualizer(args.gt_csv, args.est_txt)
    viz.show_all()

if __name__ == "__main__":
    main()
