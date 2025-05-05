from .pipeline import ImageProcessingPipeline
from .camera_model import CameraModel
from .imu_processor import IMUProcessor
from .pyramid_builder import PyramidBuilder
from .feature_meta_data import FeatureMetaData
from .feature_measurment import FeatureMeasurement
from .feature_initializer import FeatureInitializer
from .feature_adder import FeatureAdder
from .feature_tracker import FeatureTracker
from .feature_pruner import FeaturePruner
from .stereo_matcher import StereoMatcher
from .feature_publisher import FeaturePublisher

class ImageProcessor(ImageProcessingPipeline):
    """
    Facade preserving legacy API for image processing.

    Inherits:
      - ImageProcessingPipeline (implements config-based setup, imu_callback, stereo_callback)

    Provides:
      - stareo_callback alias for backward compatibility
    """
    def __init__(self, config):
        super().__init__(config)

    stareo_callback = ImageProcessingPipeline.stereo_callback
