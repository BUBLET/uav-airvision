import cv2

class FeatureMatcher:
    def __init__(self):
        """
        Инициализация объекта FeatureMatcher с использованием BFMatcher.
        """
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(self, descriptors1, descriptors2):
        """
        Сопоставляет дескрипторы между двумя изображениями.

        
        Возвращает:
        - matches (list): список сопоставленных точек.
        """
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Дескрипторы не могут быть пустыми.")
        
        # Находим соответствия
        matches = self.matcher.match(descriptors1, descriptors2)
        
        # Сортируем соответствия по расстоянию
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches

    def draw_matches(self, img1, img2, keypoints1, keypoints2, matches):
        """
        Отображает совпадения между двумя изображениями.
        
        Возвращает:
        - result_image (numpy.ndarray): изображение с отображенными совпадениями.
        """
        # Отображаем совпадения
        result_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return result_image
