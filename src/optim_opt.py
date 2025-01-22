import optuna
import subprocess
import numpy as np
from scipy.signal import correlate

# Пути
CONFIG_PATH = "./src/config.py"
TRUE_TRAJECTORY_PATH = "./results/cam_traj_truth.txt"
OUTPUT_TRAJECTORY_PATH = "./results/estimated_traj.txt"

import logging

# Настройка логирования
logging.basicConfig(
    filename="optuna_study.log",  # Имя файла для логов
    level=logging.INFO,          # Уровень логирования
    format="%(asctime)s - %(message)s",
    filemode="w",                # Перезаписывать файл при каждом запуске
)

# Callback для записи результатов каждой итерации
def log_study_results(study, trial):
    logging.info(f"Trial {trial.number}: Value = {trial.value}")
    logging.info(f"Params: {trial.params}")

# Функция для изменения параметров в config.py
def update_config(params):
    with open(CONFIG_PATH, "r") as file:
        config_lines = file.readlines()
    
    # Изменяем параметры
    for key, value in params.items():
        config_lines = [line if not line.startswith(key) else f"{key} = {value}\n" for line in config_lines]
    
    with open(CONFIG_PATH, "w") as file:
        file.writelines(config_lines)

# Нормализация траектории
def normalize_trajectory(trajectory):
    trajectory -= trajectory[0]  # Центрирование относительно начала координат
    scale = np.linalg.norm(trajectory, axis=1).max()  # Нормализация масштаба
    trajectory /= scale
    return trajectory

# Вычисление сходства с использованием кросс-корреляции
def compute_similarity(true_trajectory, output_trajectory):
    correlation = 0
    for i in range(true_trajectory.shape[1]):  # Сравнение по каждой оси
        corr = correlate(true_trajectory[:, i], output_trajectory[:, i], mode='valid')
        correlation += np.max(corr)  # Берём максимальное значение корреляции
    return correlation

# Нормализация траектории (только координаты)
def normalize_trajectory(trajectory):
    trajectory = trajectory[:, :3]  # Берём только координаты
    trajectory -= trajectory[0]  # Центрирование относительно начала координат
    scale = np.linalg.norm(trajectory, axis=1).max()  # Нормализация масштаба
    trajectory /= scale
    return trajectory

# Функция для запуска системы и вычисления ошибки
def evaluate_config(params):
    try:
        update_config(params)
        # Запуск системы
        subprocess.run(["python", "./src/main.py"], check=True)
        
        # Сравнение траекторий
        true_trajectory = np.loadtxt(TRUE_TRAJECTORY_PATH)
        output_trajectory = np.loadtxt(OUTPUT_TRAJECTORY_PATH)

        # Нормализация траекторий
        true_trajectory = normalize_trajectory(true_trajectory)
        output_trajectory = normalize_trajectory(output_trajectory)

        # Приведение к одинаковой длине для корреляции
        min_length = min(len(true_trajectory), len(output_trajectory))
        true_trajectory = true_trajectory[:min_length]
        output_trajectory = output_trajectory[:min_length]

        # Вычисление корреляции
        similarity = compute_similarity(true_trajectory, output_trajectory)
        return -similarity  # Оптимизация должна минимизировать значение

    except subprocess.CalledProcessError:
        # Если subprocess завершился с ошибкой, возвращаем большой штраф
        return float("inf")
    except Exception as e:
        # Логирование прочих ошибок
        print(f"Ошибка при вычислении: {e}")
        return float("inf")


# Оптимизация
def objective(trial):
    params = {
        "LOWE_RATIO": trial.suggest_float("LOWE_RATIO", 0.5, 0.9),
        "E_RANSAC_THRESHOLD": trial.suggest_float("E_RANSAC_THRESHOLD", 0.05, 0.5),
        "TRIANGULATION_THRESHOLD_DEG": trial.suggest_float("TRIANGULATION_THRESHOLD_DEG", 0.5, 2.0),
        "ROTATION_THRESHOLD_DEG": trial.suggest_float("ROTATION_THRESHOLD_DEG", 0.5, 2.0),
        "TRANSLATION_THRESHOLD": trial.suggest_float("TRANSLATION_THRESHOLD", 0.05, 1.0),
        "EPIPOLAR_THRESHOLD": trial.suggest_float("EPIPOLAR_THRESHOLD", 0.01, 1),
        "FORCE_KEYFRAME_INTERVAL": trial.suggest_int("FORCE_KEYFRAME_INTERVAL", 1, 15),
        "KEYFRAME_BA_INTERVAL": trial.suggest_int("KEYFRAME_BA_INTERVAL", 1, 5),
        "MIN_OBSERVATIONS": trial.suggest_int("MIN_OBSERVATIONS", 1, 4),
        "BUNDLE_ADJUSTMENT_FRAMES": trial.suggest_int("BUNDLE_ADJUSTMENT_FRAMES", 3, 10),
        "MAX_REPROJ_ERROR": trial.suggest_float("MAX_REPROJ_ERROR", 50, 150),
        "MAP_CLEAN_MAX_DISTANCE": trial.suggest_float("MAP_CLEAN_MAX_DISTANCE", 10, 200),
        "KPTS_UNIFORM_SELECTION_GRID_SIZE": trial.suggest_int("KPTS_UNIFORM_SELECTION_GRID_SIZE", 8, 32),
        "MAX_PTS_PER_GRID": trial.suggest_int("MAX_PTS_PER_GRID", 4, 16),
        "BA_FTOL": trial.suggest_float("BA_FTOL", 0.0001, 0.01),
        "BA_XTOL": trial.suggest_float("BA_XTOL", 0.0001, 0.01),
        "BA_GTOL": trial.suggest_float("BA_GTOL", 0.0001, 0.01)
    }
    return evaluate_config(params)

# Увеличиваем число итераций
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000, callbacks=[log_study_results])  # 200 итераций для большей точности

print("Лучшие параметры:", study.best_params)
print("Максимальная корреляция (с минимальным отрицательным значением):", -study.best_value)
