import cv2
import numpy as np
from pathlib import Path

from config import DATASET_PATH, OUTPUT_TRAJ
from image_processing.frame_processor import FrameProcessor

class OdometryPipeline:
    def __init__(
        self,
        cam_df,
        processor: FrameProcessor,
        viz_show_func,
        viz_save_func
    ):
        """
        :param cam_df: DataFrame с метками времени кадров
        :param processor: экземпляр FrameProcessor
        :param viz_show_func: функция отображения кадра (viz.show)
        :param viz_save_func: функция сохранения траектории (viz.save)
        """
        self.cam_df = cam_df
        self.processor = processor
        self.show = viz_show_func
        self.save = viz_save_func

    def run(self):
        pred_states, gt_states = [], []
        pred_positions, gt_positions = [], []

        for idx, row in self.cam_df.iterrows():
            ts = int(row['#timestamp [ns]'])
            img_path = Path(DATASET_PATH) / 'cam0' / 'data' / f"{ts}.png"
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            result = self.processor.process(idx, ts, img)
            if result is None:
                continue

            frame, pred_state, gt_state, pred_pos, gt_pos = result

            pred_states.append(pred_state)
            gt_states.append(gt_state)
            pred_positions.append(pred_pos)
            gt_positions.append(gt_pos)

            if not self.show(frame):
                break

        # сохранение
        #self.save(OUTPUT_TRAJ)
        cv2.destroyAllWindows()

        return (
            np.array(pred_states),
            np.array(gt_states),
            np.array(pred_positions),
            np.array(gt_positions)
        )
