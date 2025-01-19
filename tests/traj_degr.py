#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def parse_trajectory_file(file_path):
    """
    Считывает файл траектории, где каждая строка содержит 12 чисел:
    tx, ty, tz, R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1], R[0,2], R[1,2], R[2,2].
    
    Возвращает:
    positions (list of np.ndarray, shape = (N, 3)):
        Точки (x, y, z) на каждом кадре.
    rotations (list of np.ndarray, shape = (N, 3, 3)):
        Матрицы вращения 3x3 на каждом кадре.
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
                # Можно пропустить строку или выдать предупреждение
                continue

            vals = list(map(float, vals))
            # Извлекаем трансляцию
            tx, ty, tz = vals[0], vals[1], vals[2]
            # Извлекаем элементы вращения (по столбцам):
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


def compute_position_errors(gt_positions, est_positions):
    """
    Вычисляет разницу в трансляциях (L2-норму) для каждого кадра:
    error_i = ||gt_positions[i] - est_positions[i]||
    
    Возвращает:
    errors (np.ndarray): вектор ошибок по кадрам,
    mean_error, max_error: средняя и максимальная ошибка
    """
    n = min(len(gt_positions), len(est_positions))
    if n == 0:
        return None, None, None

    # Усечём до одинакового размера, если длины отличаются
    gt_positions = gt_positions[:n]
    est_positions = est_positions[:n]

    diffs = gt_positions - est_positions  # shape (n, 3)
    errors = np.linalg.norm(diffs, axis=1)  # L2-норма покадрово

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    return errors, mean_error, max_error


def compute_percentage_error(gt_positions, est_positions):
    """
    Пример простой процентной ошибки:
    % = (||est - gt|| / ||gt||) * 100%
    для каждого кадра.
    Если gt ~ 0, то процентная ошибка условно = 0 или игнорируем.
    """
    n = min(len(gt_positions), len(est_positions))
    if n == 0:
        return None

    gt_positions = gt_positions[:n]
    est_positions = est_positions[:n]
    gt_norm = np.linalg.norm(gt_positions, axis=1)  # длина вектора (x,y,z)
    est_norm = np.linalg.norm(est_positions, axis=1)

    # Можно считать процент разницы норм
    # напр. perc[i] = (|est_norm[i] - gt_norm[i]| / gt_norm[i]) * 100
    # или использовать l2-норму (est_i - gt_i) / l2(gt_i).

    percentages = []
    for i in range(n):
        denom = gt_norm[i]
        if denom < 1e-9:
            percentages.append(0.0)
        else:
            num = abs(est_norm[i] - denom)
            percentages.append((num / denom) * 100.0)

    return np.array(percentages)


def plot_trajectory_comparison(gt_positions, est_positions, errors=None):
    """
    Рисует на одном графике (x, y, z) координаты истинной и оценённой траекторий.
    Если переданы errors (покадровые), рисует их на отдельном subplot.
    """
    n = min(len(gt_positions), len(est_positions))
    frames = np.arange(n)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    # Подграфик (x)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(frames, gt_positions[:n, 0], label="GT X", c='blue')
    ax1.plot(frames, est_positions[:n, 0], label="Est X", c='red')
    ax1.set_title("Coordinate X")
    ax1.legend()

    # (y)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(frames, gt_positions[:n, 1], label="GT Y", c='blue')
    ax2.plot(frames, est_positions[:n, 1], label="Est Y", c='red')
    ax2.set_title("Coordinate Y")

    # (z)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(frames, gt_positions[:n, 2], label="GT Z", c='blue')
    ax3.plot(frames, est_positions[:n, 2], label="Est Z", c='red')
    ax3.set_title("Coordinate Z")

    # Если есть ошибки по кадрам, нарисуем их
    if errors is not None:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(frames, errors, label="Position Error (L2)", c='magenta')
        ax4.set_title("Positional Error per Frame")
        ax4.legend()
    else:
        # Скрыть оси
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Путь к файлам с траекториями (если у вас другая структура — меняйте)
    gt_file = os.path.join("results", "cam_traj_truth.txt")
    est_file = os.path.join("results", "estimated_traj.txt")

    # Парсим обе траектории
    gt_positions, gt_rotations = parse_trajectory_file(gt_file)
    est_positions, est_rotations = parse_trajectory_file(est_file)

    # Считаем ошибки по трансляции
    errors, mean_error, max_error = compute_position_errors(gt_positions, est_positions)

    # Считаем условную процентную ошибку (один из вариантов)
    perc_errors = compute_percentage_error(gt_positions, est_positions)

    # Печатаем краткие итоги
    n = min(len(gt_positions), len(est_positions))
    print(f"Number of frames compared: {n}")
    if errors is not None:
        print(f"Mean position error (L2): {mean_error:.4f}")
        print(f"Max position error (L2): {max_error:.4f}")
    if perc_errors is not None:
        mean_perc = np.mean(perc_errors)
        print(f"Mean percentage error: {mean_perc:.2f}%")
    
    # Визуализируем позиции и ошибки
    plot_trajectory_comparison(gt_positions, est_positions, errors)


if __name__ == "__main__":
    main()
