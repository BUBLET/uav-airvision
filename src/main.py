import cv2
from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher
from image_processing.odometry_calculation import OdometryCalculator

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
    focal_length = 800.0  # Фокусное расстояние (800 вроде как стандарт нужна инфа с камеры)
    principal_point = (0.5, 0.5)  # Нужна инфа с камеры, либо середина
    odometry_calculator = OdometryCalculator(focal_length, principal_point)

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
        
        # Если кадр не был прочитан выходим из цикла
        if not ret:
            break
        
        # Извлекаем ключевые точки и дескрипторы для текущего кадра
        keypoints, descriptors = feature_extractor.extract_features(frame)
        
        # Сопоставляем ключевые точки между предыдущим и текущим кадрами
        matches = feature_matcher.match_features(prev_descriptors, descriptors)

        # Вычисляем движение с помощью одометрии
        translation, rotation = odometry_calculator.calculate_motion(prev_keypoints, keypoints, matches)
        
        
        # Визуализируем вектор смещения
        start_point = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))  # Центр изображения
        scale = 50  # Масштаб вектора
        end_point = (int(start_point[0] + translation[0] * scale),
                     int(start_point[1] - translation[1] * scale))  # Инвертируем Y для корректного отображени
        
         # Рисуем вектор смещения
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.1)

        # Отображаем совпадения между кадрами
        frame_with_matches = feature_matcher.draw_matches(prev_frame, frame, prev_keypoints, keypoints, matches)
        
        # Показ кадра с совпадениями
        cv2.imshow("Matches", frame_with_matches)

        # Выводим информацию о движении
        print(f"Translation: {translation.flatten()}, Rotation: {rotation}")
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
