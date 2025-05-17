import cv2
import numpy as np

class PyramidBuilder:
    def __init__(self, win_size, pyramid_levels,
                 cam0_curr_img_msg, cam1_curr_img_msg):
        """
        win_size:         кортеж или число — размер окна для LK-метода
        pyramid_levels:   число уровней пирамиды
        cam0_curr_img_msg: сообщение с текущим изображением левой камеры
        cam1_curr_img_msg: сообщение с текущим изображением правой камеры
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
        Строит (или использует оригинал) пирамиды изображений для обеих камер.
        
        Возвращает:
            pyr0: пирамида или изображение для cam0
            pyr1: пирамида или изображение для cam1
        """
        img0 = self.cam0_curr_img_msg.image
        # анкоммент, если buildOpticalFlowPyramid работает:
        # pyr0 = cv2.buildOpticalFlowPyramid(
        #     img0, self.win_size, self.pyramid_levels,
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False
        # )[1]
        pyr0 = img0

        img1 = self.cam1_curr_img_msg.image
        # анкоммент, если buildOpticalFlowPyramid работает:
        # pyr0 = cv2.buildOpticalFlowPyramid(
        #     img0, self.win_size, self.pyramid_levels,
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False
        # )[1]
        pyr1 = img1

        self.curr_cam0_pyramid = pyr0
        self.curr_cam1_pyramid = pyr1
        return pyr0, pyr1
