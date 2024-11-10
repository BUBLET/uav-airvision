import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

class DepthEstimator:
    def __init__(self):
        # Загрузка модели MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
        self.model.eval()
        
        # Предобработка для модели
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Проверка, доступен ли GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def estimate_depth(self, frame):
        # Преобразование BGR в RGB
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Преобразуем изображение в тензор
        input_tensor = self.transform(input_frame).unsqueeze(0).to(self.device)
        
        # Оценка глубины
        with torch.no_grad():
            depth_map = self.model(input_tensor)
        
        # Преобразование к нужному формату
        depth_map = depth_map.squeeze().cpu().numpy()  # Убираем лишнюю размерность
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)  # Нормализуем
        depth_map = depth_map.astype("uint8")  # Преобразуем в uint8
        
        return depth_map
