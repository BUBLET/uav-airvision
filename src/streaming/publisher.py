import numpy as np
import cv2
import time

from threading import Thread
from .dataset import EuRoCDataset

class DataPublisher(object):
    def __init__(self, dataset, out_queue, duration=float('inf'), ratio=1.): 
        self.dataset = dataset
        self.dataset_starttime = dataset.starttime
        self.out_queue = out_queue
        self.duration = duration
        self.ratio = ratio
        self.starttime = None
        self.started = False
        self.stopped = False

        self.publish_thread = Thread(target=self.publish)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()

    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
        self.out_queue.put(None)

    def publish(self):
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                data = next(dataset)
            except StopIteration:
                self.out_queue.put(None)
                return

            interval = data.timestamp - self.dataset_starttime
            if interval < 0:
                continue
            while (time.time() - self.starttime) * self.ratio < interval + 1e-3:
                time.sleep(1e-3)   # assumption: data frequency < 1000hz
                if self.stopped:
                    return

            if interval <= self.duration + 1e-3:
                self.out_queue.put(data)
            else:
                self.out_queue.put(None)
                return



if __name__ == '__main__':
    from queue import Queue

    path = 'path/to/your/EuRoC Mav Dataset/MH_01_easy'
    dataset = EuRoCDataset(path)
    dataset.set_starttime(offset=30)

    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()

    duration = 1
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration)
    gt_publisher = DataPublisher(
        dataset.groundtruth, gt_queue, duration)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    # gt_publisher.start(now)

    def print_msg(in_queue, source):
        while True:
            x = in_queue.get()
            if x is None:
                return
            print(x.timestamp, source)
    t2 = Thread(target=print_msg, args=(imu_queue, 'imu'))
    t3 = Thread(target=print_msg, args=(gt_queue, 'groundtruth'))
    t2.start()
    t3.start()

    timestamps = []
    while True:
        x = img_queue.get()
        if x is None:
            break
        print(x.timestamp, 'image')
        cv2.imshow('left', np.hstack([x.cam0_image, x.cam1_image]))
        cv2.waitKey(1)
        timestamps.append(x.timestamp)

    imu_publisher.stop()
    img_publisher.stop()
    gt_publisher.stop()
    t2.join()
    t3.join()

    print(f'\nelapsed time: {time.time() - now}s')
    print(f'dataset time interval: {timestamps[-1]} -> {timestamps[0]}'
        f'  ({timestamps[-1]-timestamps[0]}s)\n')
    print('Please check if IMU and image are synced')