import os
import numpy as np

class TrajectoryWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.file = open(output_path, 'w')
        # Записываем header, похожий на GT
        self.file.write( 
            "#timestamp [s],p_x [m],p_y [m],p_z [m]," +
            "R11,R21,R31,R12,R22,R32,R13,R23,R33\n"
        )

    def write_pose(self, timestamp: float, t: np.ndarray, R: np.ndarray):
        """
        timestamp: сек., t: (3,1) или (3,), R: (3,3)
        """
        t = t.flatten()
        # R в столбцовом порядке (так же как flatten('F'))
        r = R.flatten('F')
        data = np.hstack(([timestamp], t, r))
        line = ",".join(f"{v:.6f}" for v in data)
        self.file.write(line + "\n")

    def close(self):
        self.file.close()
