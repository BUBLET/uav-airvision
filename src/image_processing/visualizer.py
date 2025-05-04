import cv2
import numpy as np

class TrajectoryVisualizer:
    def __init__(self, size=400, scale=25, window_name='VO + GT Trajectories'):
        self.size = size
        self.center = size // 2
        self.scale = scale
        self.window_name = window_name
        self.reset_canvas()

    def reset_canvas(self):
        self.traj = {
            'xy': np.zeros((self.size, self.size, 3), dtype=np.uint8),
            'xz': np.zeros((self.size, self.size, 3), dtype=np.uint8),
            'yz': np.zeros((self.size, self.size, 3), dtype=np.uint8),
        }
        self.init_pos = None

    def update(self, t, true_vals):
        if self.init_pos is None:
            self.init_pos = t.copy()

        def draw(a1, a2, canvas):
            x1 = int((t[a1] - self.init_pos[a1]) * self.scale) + self.center
            y1 = int((t[a2] - self.init_pos[a2]) * self.scale) + self.center
            x2 = int((true_vals[a1] - self.init_pos[a1]) * self.scale) + self.center
            y2 = int((true_vals[a2] - self.init_pos[a2]) * self.scale) + self.center
            if 0 <= x1 < self.size and 0 <= y1 < self.size:
                cv2.circle(canvas, (x1, y1), 1, (0, 255, 0), 1)
            if 0 <= x2 < self.size and 0 <= y2 < self.size:
                cv2.circle(canvas, (x2, y2), 1, (0, 0, 255), 2)

        draw(0, 1, self.traj['xy'])
        draw(0, 2, self.traj['xz'])
        draw(1, 2, self.traj['yz'])

    def compose_frame(self, img_gray):
        cam_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cam_vis = cv2.resize(cam_vis, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        top = np.hstack((self.traj['xy'], self.traj['xz']))
        bottom = np.hstack((self.traj['yz'], cam_vis))
        return np.vstack((top, bottom))

    def show(self, frame):
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    def save(self, output_paths):
        for key, path in output_paths.items():
            cv2.imwrite(str(path), self.traj[key])