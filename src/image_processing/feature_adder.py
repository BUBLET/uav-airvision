import numpy as np
import cv2

from itertools import chain
from .feature_meta_data import FeatureMetaData

class FeatureAdder:
    def __init__(self,
                 detector,
                 stereo_matcher,
                 config,
                 cam0_curr_img_msg,
                 curr_features,
                 next_feature_id,
                 grid_row,
                 grid_col,
                 grid_max_feature_num,
                 grid_min_feature_num):
        """
        detector:               FastFeatureDetector
        stereo_matcher:         StereoMatcher
        config:                 конфиг с grid_num и др.
        cam0_curr_img_msg:      текущее сообщение с изображением
        curr_features:          список [[FeatureMetaData]] для заполнения
        next_feature_id:        счётчик для присвоения id новым фичам
        grid_row/col:           разбивка на сетку
        grid_max_feature_num:   верхний предел точек
        grid_min_feature_num:   нижний предел точек
        """
        self.detector          = detector
        self.stereo_matcher    = stereo_matcher
        self.stereo_match      = stereo_matcher.stereo_match

        self.config            = config
        self.cam0_curr_img_msg = cam0_curr_img_msg
        self.curr_features     = curr_features
        self.next_feature_id   = next_feature_id

        self.grid_row          = grid_row
        self.grid_col          = grid_col
        self.grid_max_feature_num = grid_max_feature_num
        self.grid_min_feature_num = grid_min_feature_num

    def get_grid_size(self, img):
        """
        Size of each grid.
        """
        grid_height = int(np.ceil(img.shape[0] / self.grid_row))
        grid_width  = int(np.ceil(img.shape[1] / self.grid_col))
        return grid_height, grid_width
    
    def add_new_features(self):
        """
        Detect new features on the image to ensure that the features are 
        uniformly distributed on the image.
        """
        curr_img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(curr_img)

        # Create a mask to avoid redetecting existing features.
        mask = np.ones(curr_img.shape[:2], dtype='uint8')
        for feature in chain.from_iterable(self.curr_features):
            x, y = map(int, feature.cam0_point)
            mask[y-3:y+4, x-3:x+4] = 0

        # Detect new features.
        new_features = self.detector.detect(curr_img, mask=mask)

        # Collect the new detected features based on the grid.
        new_feature_sieve = [[] for _ in range(self.config.grid_num)]
        for kp in new_features:
            row = int(kp.pt[1] / grid_height)
            col = int(kp.pt[0] / grid_width)
            code = row * self.grid_col + col
            new_feature_sieve[code].append(kp)

        new_features = []
        for cell in new_feature_sieve:
            if len(cell) > self.grid_max_feature_num:
                cell = sorted(cell, key=lambda x: x.response, reverse=True)[:self.grid_max_feature_num]
            new_features.extend(cell)

        # Stereo match
        cam0_points = [kp.pt for kp in new_features]
        cam1_points, inlier_markers = self.stereo_match(cam0_points)

        # Filter inliers
        cam0_inliers, cam1_inliers, response_inliers = [], [], []
        for i, ok in enumerate(inlier_markers):
            if not ok:
                continue
            cam0_inliers.append(cam0_points[i])
            cam1_inliers.append(cam1_points[i])
            response_inliers.append(new_features[i].response)

        # Group by grid
        grid_new_features = [[] for _ in range(self.config.grid_num)]
        for pt0, pt1, resp in zip(cam0_inliers, cam1_inliers, response_inliers):
            row = int(pt0[1] / grid_height)
            col = int(pt0[0] / grid_width)
            code = row * self.grid_col + col

            fm = FeatureMetaData()
            fm.response   = resp
            fm.cam0_point = pt0
            fm.cam1_point = pt1
            grid_new_features[code].append(fm)

        # Select top in each grid
        for i, feats in enumerate(grid_new_features):
            top = sorted(feats, key=lambda x: x.response, reverse=True)[:self.grid_min_feature_num]
            for f in top:
                f.id       = self.next_feature_id
                f.lifetime = 1
                self.curr_features[i].append(f)
                self.next_feature_id += 1
