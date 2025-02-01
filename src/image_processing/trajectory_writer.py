import os
import numpy as np

class TrajectoryWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.file = self._initialize_file()

    def _initialize_file(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        return open(self.output_path, 'w')

    def write_pose(self, t: np.ndarray, R: np.ndarray):
        t = t.flatten()
        R = R.flatten('F')
        pose = np.hstack((t, R))
        line = ' '.join(f'{num:.6f}' for num in pose)
        self.file.write(line + '\n')

    def close(self):
        if not self.file.closed:
            self.file.close()
