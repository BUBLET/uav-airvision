import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator
from error_correction.error_correction import ErrorCorrector

def main():
    # Загружаем видеофайл
    video_path = "datasets/video1.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка: не удалось загрузить видео.")
        return
    
    # Создаем объект FeatureExtractor и FeatureMatcher
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    
    # Параметры камеры
    focal_length = 800.0  # Фокусное расстояние
    principal_point = (0.5, 0.5)  # Опорная точка
    odometry_calculator = OdometryCalculator(focal_length, principal_point)

    # Параметры фильтра Калмана
    dt = 0.1  # Шаг времени между кадрами
    process_noise = 1e-3
    measurement_noise = 1e-2
    error_corrector = ErrorCorrector(dt, process_noise, measurement_noise)

    # Параметр сглаживания
    alpha = 0.8  # Чем меньше значение, тем сильнее сглаживание

    # Инициализация накопленного смещения
    accumulated_translation = np.zeros(2)

    # Списки для хранения координат траектории
    trajectory_x = []
    trajectory_y = []

    # Читаем первый кадр
    ret, prev_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        return
    
    # Извлекаем ключевые точки и дескрипторы для первого кадра
    prev_keypoints, prev_descriptors = feature_extractor.extract_features(prev_frame)
    
    while True:
        # Читаем следующий кадр из видео
        ret, frame = cap.read()
        
        # Если кадр не был прочитан, выходим из цикла
        if not ret:
            break
        
        # Извлекаем ключевые точки и дескрипторы для текущего кадра
        keypoints, descriptors = feature_extractor.extract_features(frame)
        
        # Сопоставляем ключевые точки между предыдущим и текущим кадром
        matches = feature_matcher.match_features(prev_descriptors, descriptors)

        # Визуализируем сопоставленные ключевые точки
        matched_frame = cv2.drawMatches(prev_frame, prev_keypoints, frame, keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matched_frame_resized = cv2.resize(matched_frame, (1200, 600))
        
        # Вычисляем движение с помощью одометрии
        translation, rotation = odometry_calculator.calculate_motion(prev_keypoints, keypoints, matches)
        
        # Выводим исходное смещение для диагностики
        print(f"Translation (до фильтрации): {translation.flatten()}")
        
        # Применяем фильтр Калмана для коррекции смещения
        corrected_position, corrected_velocity = error_corrector.apply_correction(translation.flatten())
        
        # Выводим исправленное смещение для диагностики
        print(f"Translation (после фильтрации): {corrected_position}")
        
        # Обновляем накопленное смещение с учетом текущего смещения
        accumulated_translation = (1 - alpha) * accumulated_translation + alpha * corrected_position[:2]

        # Добавляем новые координаты в траекторию
        trajectory_x.append(accumulated_translation[0])
        trajectory_y.append(-accumulated_translation[1])  # Инвертируем ось Y, чтобы траектория была в правильном направлении

        # Отображаем кадр с визуализацией ключевых точек
        cv2.imshow("Matched Keypoints", matched_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Обновляем предыдущий кадр и его ключевые точки и дескрипторы
        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors

        # Отрисовываем текущую траекторию на графике
        plt.clf()  # Очищаем предыдущий график
        plt.plot(trajectory_x, trajectory_y, marker='o', markersize=3, linestyle='-', color='b')
        plt.title("2D Траектория движения камеры")
        plt.xlabel("X (Горизонтальная ось)")
        plt.ylabel("Y (Вертикальная ось)")
        plt.grid(True)
        plt.pause(0.1)  # Небольшая пауза, чтобы обновить график

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
