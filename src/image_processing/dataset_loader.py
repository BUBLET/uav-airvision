import os
import cv2
import glob
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

class BaseDatasetLoader(ABC):
    @abstractmethod
    def read_next(self) -> Optional[Tuple[cv2.Mat, float]]:
        pass
    def release(self):
        pass


class VideoLoader(BaseDatasetLoader):
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

    def read_next(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame, None

    def release(self):
        self.cap.release()


class FramesLoader(BaseDatasetLoader):
    def __init__(self, folder: str, ext: str = "png"):
        self.paths = sorted(glob.glob(os.path.join(folder, f"*.{ext}")))
        if not self.paths:
            raise ValueError(f"Не найдено файлов {ext} в {folder}")
        self.idx = 0

    def read_next(self):
        if self.idx >= len(self.paths):
            return None
        frame = cv2.imread(self.paths[self.idx])
        if frame is None:
            raise ValueError(f"Не удалось прочитать {self.paths[self.idx]}")
        self.idx += 1
        return frame, None


class TUMVILoader(BaseDatasetLoader):
    def __init__(self, root: str, cam: str = "cam0"):
        csv_path = os.path.join(root, cam, "data.csv")
        if not os.path.isfile(csv_path):
            raise ValueError(f"TUM VI: не найден {csv_path}")
        # используем delim_whitespace, чтобы pandas правильно разбил колонки
        df = pd.read_csv(
            csv_path,
            delim_whitespace=True,
            comment='#',
            header=None,
            names=["ts", "file"]
        )
        # приводим ts к целому
        df["ts"] = df["ts"].astype(np.int64)
        self.entries = [
            (int(row.ts) * 1e-9, os.path.join(root, cam, row.file))
            for row in df.itertuples()
        ]
        self.idx = 0

    def read_next(self):
        if self.idx >= len(self.entries):
            return None
        ts, path = self.entries[self.idx]
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Не удалось прочитать {path}")
        self.idx += 1
        return frame, ts


class EuRoCLoader(BaseDatasetLoader):
    def __init__(self, root: str, cam: str = "cam0"):
        base = os.path.join(root, "mav0", cam)
        csv_path = os.path.join(base, "data.csv")
        if not os.path.isfile(csv_path):
            raise ValueError(f"EuRoC: не найден {csv_path}")

        # Пропускаем первую строку-заголовок, затем разбиваем по запятой
        df = pd.read_csv(
            csv_path,
            sep=',',
            skiprows=1,
            header=None,
            names=["ts", "file"],
            dtype={"ts": np.int64, "file": str}
        )

        self.entries = []
        for row in df.itertuples(index=False):
            ts = row.ts * 1e-9
            rel = row.file  # e.g. "1413393212255760384.png"
            # сначала пробуем без папки data/, потом внутри data/
            path1 = os.path.join(base, rel)
            if os.path.isfile(path1):
                full_path = path1
            else:
                full_path = os.path.join(base, "data", rel)
                if not os.path.isfile(full_path):
                    raise ValueError(f"EuRoC: не найден файл {path1} или {full_path}")
            self.entries.append((ts, full_path))

        self.idx = 0

    def read_next(self):
        if self.idx >= len(self.entries):
            return None
        ts, path = self.entries[self.idx]
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Не удалось прочитать {path}")
        self.idx += 1
        return frame, ts

