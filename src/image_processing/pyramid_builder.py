import cv2
import numpy as np

class PyramidBuilder:
    def __init__(self, win_size, pyramid_levels,
                 cam0_curr_img_msg, cam1_curr_img_msg):
        """
        win_size:         tuple or int for LK window size
        pyramid_levels:   number of pyramid levels
        cam0_curr_img_msg: current left-camera image message
        cam1_curr_img_msg: current right-camera image message
        """
        self.win_size = win_size
        self.pyramid_levels = pyramid_levels
        self.cam0_curr_img_msg = cam0_curr_img_msg
        self.cam1_curr_img_msg = cam1_curr_img_msg

        # placeholders for pyramids
        self.curr_cam0_pyramid = None
        self.curr_cam1_pyramid = None

    def create_image_pyramids(self):
        """
        Build (or fallback to raw image) pyramids for both cameras.
        Returns:
            pyr0: pyramid or image for cam0
            pyr1: pyramid or image for cam1
        """
        img0 = self.cam0_curr_img_msg.image
        # uncomment if buildOpticalFlowPyramid works:
        # pyr0 = cv2.buildOpticalFlowPyramid(
        #     img0, self.win_size, self.pyramid_levels,
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False
        # )[1]
        pyr0 = img0

        img1 = self.cam1_curr_img_msg.image
        # uncomment if buildOpticalFlowPyramid works:
        # pyr1 = cv2.buildOpticalFlowPyramid(
        #     img1, self.win_size, self.pyramid_levels,
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False
        # )[1]
        pyr1 = img1

        self.curr_cam0_pyramid = pyr0
        self.curr_cam1_pyramid = pyr1
        return pyr0, pyr1
