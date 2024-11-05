import cv2

class FeatureExtractor:
    def __init__(self):
        """
        Инициализация объекта FeatureExtractor с использованием ORB.
        """
        self.extractor = cv2.ORB_create()

    def extract_features(self, image):
        """
        Извлекает ключевые точки и дескрипторы из изображения с использованием ORB.
        
        Параметры:
        - image (numpy.ndarray): входное изображение.
        
        Возвращает:
        - keypoints (list): список ключевых точек.
        - descriptors (numpy.ndarray): массив дескрипторов.
        """
        if image is None:
            raise ValueError("Изображение не может быть пустым.")
        
        # Преобразуем изображение в градации серого, если оно цветное
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Извлекаем ключевые точки и дескрипторы
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        
        return keypoints, descriptors

    def draw_keypoints(self, image, keypoints):
        """
        Отображает ключевые точки на изображении для визуализации.
        
        Параметры:
        - image (numpy.ndarray): исходное изображение.
        - keypoints (list): список ключевых точек.
        
        Возвращает:
        - image_with_keypoints (numpy.ndarray): изображение с нанесенными ключевыми точками.
        """
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
        return image_with_keypoints


# Пример использования
if __name__ == "__main__":
    # Загружаем изображение
    image = cv2.imread("test_image.jpg")
    
    # Создаем объект FeatureExtractor
    feature_extractor = FeatureExtractor()
    
    # Извлекаем ключевые точки и дескрипторы
    keypoints, descriptors = feature_extractor.extract_features(image)
    
    # Отображаем ключевые точки на изображении
    image_with_keypoints = feature_extractor.draw_keypoints(image, keypoints)
    
    # Показ изображения с ключевыми точками
    cv2.imshow("Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
