from .base_feature import BaseFeature
from .feature_depth_estimator import FeatureDepthEstimator
from .feature_motion_checker import FeatureMotionChecker
from .feature_observation import FeatureObservation
from .feature_position_initializer import FeaturePositionInitializer

class Feature(BaseFeature,
              FeatureDepthEstimator,
              FeatureMotionChecker,
              FeatureObservation,
              FeaturePositionInitializer):
    def __init__(self, new_id=0, optimization_config=None, T_cam0_cam1=None):
        BaseFeature.__init__(self, new_id, optimization_config)
        if T_cam0_cam1 is not None:
            BaseFeature.R_cam0_cam1 = T_cam0_cam1[:3, :3]
            BaseFeature.t_cam0_cam1 = T_cam0_cam1[:3,  3]
