import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread



class GroundTruthReader(object):
    def __init__(self, path, scaler, starttime=-float('inf')):
        self.scaler = scaler   # convert timestamp from ns to second
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('gt_msg', ['p', 'q', 'v', 'bw', 'ba'])

    def parse(self, line):
        """
        line: (timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], 
        q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], 
        v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], 
        b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], 
        b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2])
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        p = np.array(line[1:4])
        q = np.array(line[4:8])
        v = np.array(line[8:11])
        bw = np.array(line[11:14])
        ba = np.array(line[14:17])
        return self.field(timestamp, p, q, v, bw, ba)

    def set_starttime(self, starttime):
        self.starttime = starttime

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data



class IMUDataReader(object):
    def __init__(self, path, scaler, starttime=-float('inf')):
        self.scaler = scaler
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('imu_msg', 
            ['timestamp', 'angular_velocity', 'linear_acceleration'])

    def parse(self, line):
        """
        line: (timestamp [ns],
        w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],  
        a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2])
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        wm = np.array(line[1:4])
        am = np.array(line[4:7])
        return self.field(timestamp, wm, am)

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        # return next(self).timestamp
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime



class ImageReader(object):
    def __init__(self, ids, timestamps, starttime=-float('inf')):
        self.ids = ids
        self.timestamps = timestamps
        self.starttime = starttime
        self.cache = dict()
        self.idx = 0

        self.field = namedtuple('img_msg', ['timestamp', 'image'])

        self.ahead = 10   # 10 images ahead of current index
        self.wait = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        return cv2.imread(path, -1)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.wait:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if self.timestamps[i] < self.starttime:
                    continue
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            if timestamp < self.starttime:
                continue
            yield self.field(timestamp, self[i])

    def start_time(self):
        return self.timestamps[0]

    def set_starttime(self, starttime):
        self.starttime = starttime



class Stereo(object):
    def __init__(self, cam0, cam1):
        self.cam0 = cam0
        self.cam1 = cam1
        self.timestamps = cam0.timestamps

        self.field = namedtuple('stereo_msg', 
            ['timestamp', 'cam0_image', 'cam1_image', 'cam0_msg', 'cam1_msg'])

    def __iter__(self):
        for l, r in zip(self.cam0, self.cam1):
            #assert abs(l.timestamp - r.timestamp) < 0.01, 'unsynced stereo pair'
            yield self.field(l.timestamp, l.image, r.image, l, r)

    def __len__(self):
        return len(self.cam0)

    def start_time(self):
        return self.cam0.starttime

    def set_starttime(self, starttime):
        self.starttime = starttime
        self.cam0.set_starttime(starttime)
        self.cam1.set_starttime(starttime)
        
    

class EuRoCDataset(object):   # Stereo + IMU
    '''
    path example: 'path/to/your/EuRoC Mav Dataset/MH_01_easy'
    '''
    def __init__(self, path):
        self.groundtruth = GroundTruthReader(os.path.join(
            path, 'mav0', 'state_groundtruth_estimate0', 'data.csv'), 1e-9)
        self.imu = IMUDataReader(os.path.join(
            path, 'mav0', 'imu0', 'data.csv'), 1e-9)
        self.cam0 = ImageReader(
            *self.list_imgs(os.path.join(path, 'mav0', 'cam0', 'data')))
        self.cam1 = ImageReader(
            *self.list_imgs(os.path.join(path, 'mav0', 'cam1', 'data')))

        self.stereo = Stereo(self.cam0, self.cam1)
        self.timestamps = self.cam0.timestamps

        self.starttime = max(self.imu.start_time(), self.stereo.start_time())
        self.set_starttime(0)

    def set_starttime(self, offset):
        self.groundtruth.set_starttime(self.starttime + offset)
        self.imu.set_starttime(self.starttime + offset)
        self.cam0.set_starttime(self.starttime + offset)
        self.cam1.set_starttime(self.starttime + offset)
        self.stereo.set_starttime(self.starttime + offset)

    def list_imgs(self, dir):
        xs = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        xs = sorted(xs, key=lambda x:float(x[:-4]))
        timestamps = [float(_[:-4]) * 1e-9 for _ in xs]
        return [os.path.join(dir, _) for _ in xs], timestamps