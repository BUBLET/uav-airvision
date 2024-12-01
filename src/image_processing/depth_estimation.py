import os
import torch
import numpy as np
from typing import Optional

class DepthEstimator:
    def __init__(self):
        self.model_type = "DPT_Large"
        self.model_path = f"{self.model_type}_model.pt"
        self.model = self._load_model()
        self.transform = self._load_transform()

    def _load_model(self) -> torch.nn.Module:
        # Проверка, существует ли модель в локальном хранилище
        if os.path.exists(self.model_path):
            print("Загружаем модель из файла...")
            return torch.load(self.model_path)
        else:
            print("Загружаем модель с сервера и сохраняем локально...")
            model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            torch.save(model, self.model_path)
            return model

    def _load_transform(self):
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            return midas_transforms.dpt_transform
        else:
            return midas_transforms.small_transform
    
    def get_depth_frame(self, retval: bool, frame: np.ndarray) -> Optional[np.ndarray]:
        
        if not retval:
            return None
        
        input_batch = self.transform(frame)
        
        with torch.no_grad():
            predict = self.model(input_batch)

            predict = torch.nn.functional.interpolate(
                predict.unsqueeze(1),
                size = frame.shape[:2], 
                mode = "bilinear",
                align_corners=False
            ).squeeze()

            predict = predict - predict.min()
            predict = predict / predict.max()

            output = predict.cpu().numpy()
            
        return output
