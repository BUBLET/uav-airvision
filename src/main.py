import cv2
from image_processing.feature_extraction import FeatureExtractor
from image_processing.feature_matching import FeatureMatcher

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
        
        # Если кадр не был прочитан (например, видео закончилось), выходим из цикла
        if not ret:
            break
        
        # Извлекаем ключевые точки и дескрипторы для текущего кадра
        keypoints, descriptors = feature_extractor.extract_features(frame)
        
        # Сопоставляем ключевые точки между предыдущим и текущим кадрами
        matches = feature_matcher.match_features(prev_descriptors, descriptors)
        
        # Отображаем совпадения между кадрами
        frame_with_matches = feature_matcher.draw_matches(prev_frame, frame, prev_keypoints, keypoints, matches)
        
        # Показ кадра с совпадениями
        cv2.imshow("Matches", frame_with_matches)
        
        # Выходим из цикла, если нажата клавиша "q"
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
