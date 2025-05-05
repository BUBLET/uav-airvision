import time
import argparse
from queue import Queue
from streaming.dataset import EuRoCDataset
from streaming.publisher import DataPublisher
from config import ConfigEuRoC
from modules.vio import VIO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./datasets/MH_01_easy')
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.)

    img_q, imu_q = Queue(), Queue()
    viewer = None 

    config = ConfigEuRoC()
    vio = VIO(config, img_q, imu_q, viewer)
    vio.start()

    imu_pub = DataPublisher(dataset.imu, imu_q, duration=float('inf'), ratio=0.4)
    img_pub = DataPublisher(dataset.stereo, img_q, duration=float('inf'), ratio=0.4)

    now = time.time()
    imu_pub.start(now)
    img_pub.start(now)

if __name__ == '__main__':
    main()
