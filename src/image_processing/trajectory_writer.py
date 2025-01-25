import os
import numpy as np

class TrajectoryWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.file = self._initialize_file()

    def _initialize_file(self):
        """
        Очищает файл траектории и открывает его для записи.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        return open(self.output_path, 'w')

    def write_pose(self, pose: np.ndarray):
        """
        Записывает текущую позу камеры в файл траектории.
        """
        x, y, z = pose[:3, 3]
        R_flat = pose[:3, :3].flatten()
        fout_line = f"{x} {y} {z} " + " ".join(map(str, R_flat)) + "\n"
        self.file.write(fout_line)

    def close(self):
        """
        Закрывает файл траектории.
        """
        if not self.file.closed:
            self.file.close()
