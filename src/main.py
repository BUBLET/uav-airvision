import os
import time
import argparse
from queue import Queue
from streaming.dataset import EuRoCDataset
from streaming.publisher import DataPublisher
from config import ConfigEuRoC
from viewer import SimpleViewer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',   default='./datasets/V2_03_difficult')
    parser.add_argument('--offset', type=float, default=10.)
    parser.add_argument('--view',   action='store_true')
    args = parser.parse_args()

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=args.offset)

    name = os.path.basename(os.path.normpath(args.path))
    os.environ['DATASET_NAME'] = name
    os.environ['TIME_OFFSET']  = str(int(args.offset))

    from modules.vio import VIO

    img_q, imu_q = Queue(), Queue()
    viewer = SimpleViewer() if args.view else None

    vio = VIO(ConfigEuRoC(), img_q, imu_q, viewer)
    vio.start()

    now = time.time()
    DataPublisher(dataset.imu, imu_q,      duration=float('inf'), ratio=0.4).start(now)
    DataPublisher(dataset.stereo, img_q,   duration=float('inf'), ratio=0.4).start(now)

if __name__ == '__main__':
    main()
