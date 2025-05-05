from queue import Queue
from threading import Thread
from image_processing import ImageProcessor
from msckf import MSCKF

class VIO:
    def __init__(self, config, img_queue, imu_queue, viewer=None):
        self.config = config
        self.viewer = viewer
        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self._process_img, daemon=True)
        self.imu_thread = Thread(target=self._process_imu, daemon=True)
        self.vio_thread = Thread(target=self._process_feature, daemon=True)

    def start(self):
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def _process_img(self):
        while True:
            msg = self.img_queue.get()
            if msg is None:
                self.feature_queue.put(None)
                break
            if self.viewer:
                self.viewer.update_image(msg.cam0_image)
            feat = self.image_processor.stereo_callback(msg)
            if feat:
                self.feature_queue.put(feat)

    def _process_imu(self):
        while True:
            msg = self.imu_queue.get()
            if msg is None:
                break
            self.image_processor.imu_callback(msg)
            self.msckf.imu_callback(msg)

    def _process_feature(self):
        while True:
            feat = self.feature_queue.get()
            if feat is None:
                break
            result = self.msckf.feature_callback(feat)
            if result and self.viewer:
                self.viewer.update_pose(result.cam0_pose)
