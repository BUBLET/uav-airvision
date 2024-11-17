import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator
from error_correction.error_correction import ErrorCorrector

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Задаём путь к видеофайлу
    video_path = "datasets/Surenen Pass Trail Runningq.mp4"

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Ошибка: не удалось загрузить видео.")
        return

    # Читаем первый кадр
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Ошибка: не удалось прочитать первый кадр.")
        return

    # Получаем размеры изображения
    frame_height, frame_width = prev_frame.shape[:2]

    # Инициализируем объекты для обработки
    feature_extractor = FeatureExtractor(n_features=2000)
    feature_matcher = FeatureMatcher()
    odometry_calculator = OdometryCalculator(image_width=frame_width, image_height=frame_height)

    # Параметры фильтра Калмана
    dt = 1 / cap.get(cv2.CAP_PROP_FPS)  # Шаг времени между кадрами
    process_noise = 1e-2
    measurement_noise = 1e-1
    error_corrector = ErrorCorrector(dt, process_noise, measurement_noise)

    # Инициализация накопленного смещения и ориентации
    trajectory = np.zeros((3, 1))
    rotation_matrix = np.eye(3)

    # Списки для хранения координат траектории
    trajectory_x = []
    trajectory_y = []

    # Извлекаем ключевые точки и дескрипторы для первого кадра
    prev_keypoints, prev_descriptors = feature_extractor.extract_features(prev_frame)
    if len(prev_keypoints) == 0:
        logger.error("Не удалось обнаружить ключевые точки в первом кадре.")
        return

    # Настройка интерактивного графика
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], marker='o', markersize=3, linestyle='-', color='b')
    ax.set_title("2D Траектория движения камеры")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    while True:
        # Читаем следующий кадр из видео
        ret, frame = cap.read()
        if not ret:
            logger.info("Видео обработано полностью.")
            break

        # Извлекаем ключевые точки и дескрипторы для текущего кадра
        curr_keypoints, curr_descriptors = feature_extractor.extract_features(frame)
        if len(curr_keypoints) == 0:
            logger.warning("Не удалось обнаружить ключевые точки в текущем кадре. Пропуск кадра.")
            continue

        # Сопоставляем ключевые точки между предыдущим и текущим кадром
        matches = feature_matcher.match_features(prev_descriptors, curr_descriptors)
        if len(matches) < 8:
            logger.warning(f"Недостаточно совпадений ({len(matches)}) для вычисления одометрии. Пропуск кадра.")
            continue

        # Вычисляем движение с помощью одометрии
        result = odometry_calculator.calculate_motion(prev_keypoints, curr_keypoints, matches)
        if result is None:
            logger.warning("Не удалось вычислить движение между кадрами. Пропуск кадра.")
            continue
        R, t, mask = result

        # Обновляем общую ориентацию и положение
        rotation_matrix = R @ rotation_matrix
        scaled_translation = rotation_matrix.T @ t  # Преобразуем в глобальные координаты

        # Применяем фильтр Калмана для коррекции смещения
        corrected_position, corrected_velocity = error_corrector.apply_correction(scaled_translation[:2, 0])

        # Добавляем новые координаты в траекторию
        trajectory += rotation_matrix.T @ t
        trajectory_x.append(trajectory[0, 0])
        trajectory_y.append(trajectory[2, 0])  # Используем Z для плоскости XZ

        # Обновляем график траектории
        line.set_data(trajectory_x, trajectory_y)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        # Визуализируем сопоставленные ключевые точки
        matched_frame = feature_matcher.draw_matches(prev_frame, frame, prev_keypoints, curr_keypoints, matches)
        matched_frame_resized = cv2.resize(matched_frame, (1000, 1000))
        cv2.imshow("Matched Keypoints", matched_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Обновляем предыдущий кадр и его ключевые точки и дескрипторы
        prev_frame = frame
        prev_keypoints = curr_keypoints
        prev_descriptors = curr_descriptors

    # Сохраняем траекторию в файл
    np.savetxt("trajectory.txt", np.column_stack((trajectory_x, trajectory_y)), fmt='%.6f')

    # Отображаем финальную траекторию
    plt.ioff()
    plt.show()

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
