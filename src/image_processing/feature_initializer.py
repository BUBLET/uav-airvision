import numpy as np
from itertools import chain
from .feature_meta_data import FeatureMetaData

class FeatureInitializer:
    def __init__(self,
                 detector,
                 stereo_matcher,
                 config,
                 cam0_curr_img_msg,
                 curr_features,
                 next_feature_id,
                 grid_row,
                 grid_col,
                 grid_min_feature_num):
        """
        detector:               FastFeatureDetector
        stereo_matcher:         StereoMatcher
        config:                 конфиг с grid_num и др.
        cam0_curr_img_msg:      текущее сообщение с изображением
        curr_features:          список [[FeatureMetaData]] для заполнения
        next_feature_id:        счётчик для присвоения id новым фичам
        grid_row, grid_col:     разбивка на сетку
        grid_min_feature_num:   минимальное число точек в каждой ячейке
        """
        self.detector            = detector
        self.stereo_match        = stereo_matcher.stereo_match
        self.config              = config
        self.cam0_curr_img_msg   = cam0_curr_img_msg
        self.curr_features       = curr_features
        self.next_feature_id     = next_feature_id
        self.grid_row            = grid_row
        self.grid_col            = grid_col
        self.grid_min_feature_num= grid_min_feature_num

    def get_grid_size(self, img):
        """
        Size of each grid cell in pixels.
        """
        h, w = img.shape[:2]
        grid_h = int(np.ceil(h / self.grid_row))
        grid_w = int(np.ceil(w / self.grid_col))
        return grid_h, grid_w

    def initialize_first_frame(self):
        """
        Detect and initialize features on the first stereo frame.
        """
        img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(img)

        # 1) Detect keypoints in left image
        kps = self.detector.detect(img)
        cam0_points = [kp.pt for kp in kps]

        # 2) Stereo match to right image
        cam1_points, inlier_mask = self.stereo_match(cam0_points)

        # 3) Collect inlier pairs
        cam0_inliers = []
        cam1_inliers = []
        responses    = []
        for i, ok in enumerate(inlier_mask):
            if not ok:
                continue
            cam0_inliers.append(cam0_points[i])
            cam1_inliers.append(cam1_points[i])
            responses.append(kps[i].response)

        # 4) Group into grid cells
        grid_feats = [[] for _ in range(self.config.grid_num)]
        for pt0, pt1, resp in zip(cam0_inliers, cam1_inliers, responses):
            row = int(pt0[1] / grid_height)
            col = int(pt0[0] / grid_width)
            idx = row * self.grid_col + col

            fm = FeatureMetaData()
            fm.response   = resp
            fm.cam0_point = pt0
            fm.cam1_point = pt1
            grid_feats[idx].append(fm)

        # 5) Pick top-N per cell
        for idx, feats in enumerate(grid_feats):
            top_feats = sorted(feats, key=lambda f: f.response, reverse=True)[:self.grid_min_feature_num]
            for f in top_feats:
                f.id       = self.next_feature_id
                f.lifetime = 1
                self.curr_features[idx].append(f)
                self.next_feature_id += 1
