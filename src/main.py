import cv2
import collections
import numpy as np
from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator
from error_correction.error_correction import ErrorCorrector
from image_processing.depth_estimation import DepthEstimator  # Импортируем класс для оценки глубины

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
    focal_length = 800.0  # Фокусное расстояние (800 вроде как стандарт нужно инфо с камеры)
    principal_point = (0.5, 0.5)  # Нужна инфа с камеры, либо середина
    odometry_calculator = OdometryCalculator(focal_length, principal_point)

    # Параметры фильтра Калмана
    dt = 0.1  # Шаг времени между кадрами (нужно настроить под конкретное видео)
    process_noise = 1e-3
    measurement_noise = 1e-2
    error_corrector = ErrorCorrector(dt, process_noise, measurement_noise)

    # Параметр сглаживания
    alpha = 0.8  # Чем меньше значение, тем сильнее сглаживание

    # Инициализация накопленного смещения
    accumulated_translation = np.zeros(2)

    # Инициализация DepthEstimator
    depth_estimator = DepthEstimator()

    # Счетчик кадров для обновления карты глубины раз в 5 кадров
    frame_counter = 0

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

        # Вычисляем движение с помощью одометрии
        translation, rotation = odometry_calculator.calculate_motion(prev_keypoints, keypoints, matches)
        
        # Применяем фильтр Калмана для коррекции смещения
        corrected_position, corrected_velocity = error_corrector.apply_correction(translation.flatten())
        
        # Обновляем накопленное смещение с учетом текущего смещения
        accumulated_translation = (1 - alpha) * accumulated_translation + alpha * corrected_position[:2]

        # Каждый 5-й кадр обновляем карту глубины
        if frame_counter % 5 == 0:
            depth_map = depth_estimator.estimate_depth(frame)
            # Отображаем шкалу вертикального смещения (в данном случае просто показываем среднее значение глубины)
            vertical_shift = np.mean(depth_map)
            print(f"Vertical shift (average depth): {vertical_shift}")
            
            # Рисуем шкалу вертикального смещения
            scale_height = int(vertical_shift / 10)  # Преобразуем среднее значение в высоту шкалы
            cv2.line(frame, (50, 50), (50, 50 + scale_height), (0, 0, 255), 5)
            cv2.putText(frame, f"Vertical Shift: {vertical_shift:.2f} meters", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_counter += 1

        # Визуализируем кадр
        frame_resized = cv2.resize(frame, (1200, 1000))
        cv2.imshow("Frame", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Обновляем предыдущий кадр и его ключевые точки и дескрипторы
        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
