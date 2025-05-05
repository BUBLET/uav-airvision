import cv2
from collections import defaultdict

from .imu_processor import IMUProcessor
from .pyramid_builder import PyramidBuilder
from .camera_model import CameraModel
from .stereo_matcher import StereoMatcher
from .feature_initializer import FeatureInitializer
from .feature_tracker import FeatureTracker
from .feature_adder import FeatureAdder
from .feature_pruner import FeaturePruner
from .feature_publisher import FeaturePublisher

class ImageProcessingPipeline:
    def __init__(self, config):
        self.config = config
        self.prev_cam0_msg = None
        
        # IMU handler
        self.imu_processor = IMUProcessor(
            config.T_imu_cam0, config.T_imu_cam1
        )

        # Feature detector (FAST)
        self.detector = cv2.FastFeatureDetector_create(
            config.fast_threshold
        )

        # Camera geometry
        self.camera_model = CameraModel(
            config.cam0_intrinsics,
            config.cam0_distortion_model,
            config.cam0_distortion_coeffs
        )

        # State for features
        self.next_feature_id = 0
        self.prev_features   = [[] for _ in range(config.grid_num)]
        self.curr_features   = [[] for _ in range(config.grid_num)]
        self.num_features    = defaultdict(int)

        # Track whether we've seen the first frame
        self.first_frame = True
        # Keep last pyramids around for tracking
        self.prev_pyr0 = None

    def imu_callback(self, imu_msg):
        """Feed IMU messages into the IMUProcessor."""
        self.imu_processor.imu_callback(imu_msg)

    def stereo_callback(self, stereo_msg):
        """
        Process one stereo message (cam0_msg + cam1_msg).
        Returns a FeatureMeasurement message.
        """

        self.imu_processor.cam0_prev_img_msg = self.prev_cam0_msg
        self.imu_processor.cam0_curr_img_msg = stereo_msg.cam0_msg

        cam0_msg, cam1_msg = stereo_msg.cam0_msg, stereo_msg.cam1_msg

        # 1) Build pyramids for this frame
        pyramid_builder = PyramidBuilder(
            self.config.win_size,
            self.config.pyramid_levels,
            cam0_msg,
            cam1_msg
        )
        pyr0, pyr1 = pyramid_builder.create_image_pyramids()

        # 2) Prepare stereo matcher for this frame
        stereo_matcher = StereoMatcher(
            self.config.lk_params,
            self.imu_processor,
            pyramid_builder,
            self.camera_model,
            self.config.stereo_threshold
        )

        if self.first_frame:
            # 3a) Initialize on the very first frame
            initializer = FeatureInitializer(
                detector          = self.detector,
                stereo_matcher    = stereo_matcher,
                config            = self.config,
                cam0_curr_img_msg = cam0_msg,
                curr_features     = self.curr_features,
                next_feature_id   = self.next_feature_id,
                grid_row          = self.config.grid_row,
                grid_col          = self.config.grid_col,
                grid_min_feature_num = self.config.grid_min_feature_num
            )
            initializer.initialize_first_frame()
            self.next_feature_id = initializer.next_feature_id
            self.first_frame = False

        else:
            # 3b) Track existing features
            tracker = FeatureTracker(
                lk_params           = self.config.lk_params,
                imu_processor       = self.imu_processor,
                stereo_matcher      = stereo_matcher,
                cam0_intrinsics     = self.config.cam0_intrinsics,
                cam0_distortion_model  = self.config.cam0_distortion_model,
                cam0_distortion_coeffs = self.config.cam0_distortion_coeffs,
                cam1_intrinsics     = self.config.cam1_intrinsics,
                cam1_distortion_model  = self.config.cam1_distortion_model,
                cam1_distortion_coeffs = self.config.cam1_distortion_coeffs,
                prev_cam0_pyramid   = self.prev_pyr0,
                curr_cam0_pyramid   = pyr0,
                prev_features       = self.prev_features,
                curr_features       = self.curr_features,
                num_features        = self.num_features,
                grid_row            = self.config.grid_row,
                grid_col            = self.config.grid_col,
                ransac_threshold    = self.config.ransac_threshold
            )
            tracker.track_features()

            # 4) Add new features to underfilled cells
            adder = FeatureAdder(
                detector            = self.detector,
                stereo_matcher      = stereo_matcher,
                config              = self.config,
                cam0_curr_img_msg   = cam0_msg,
                curr_features       = self.curr_features,
                next_feature_id     = self.next_feature_id,
                grid_row            = self.config.grid_row,
                grid_col            = self.config.grid_col,
                grid_max_feature_num = self.config.grid_max_feature_num,
                grid_min_feature_num = self.config.grid_min_feature_num
            )
            adder.add_new_features()
            self.next_feature_id = adder.next_feature_id

            # 5) Prune overfull cells
            pruner = FeaturePruner(self.config.grid_max_feature_num)
            pruner.curr_features = self.curr_features
            pruner.config        = self.config
            pruner.prune_features()

        # 6) Publish the resulting feature measurements
        publisher = FeaturePublisher(
            self.config.cam0_intrinsics,
            self.config.cam0_distortion_model,
            self.config.cam0_distortion_coeffs,
            self.config.cam1_intrinsics,
            self.config.cam1_distortion_model,
            self.config.cam1_distortion_coeffs
        )
        publisher.cam0_curr_img_msg = cam0_msg
        publisher.cam1_curr_img_msg = cam1_msg
        publisher.curr_features     = self.curr_features

        feature_msg = publisher.publish()

        # 7) Shift state for next frame
        self.prev_cam0_msg = cam0_msg
        self.prev_features = self.curr_features
        self.curr_features = [[] for _ in range(self.config.grid_num)]
        self.prev_pyr0     = pyr0

        return feature_msg
