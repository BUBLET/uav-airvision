class FeatureMetaData(object):
    """
    Contain necessary information of a feature for easy access.
    """
    def __init__(self):
        self.id = None           # int
        self.response = None     # float
        self.lifetime = None     # int
        self.cam0_point = None   # vec2
        self.cam1_point = None   # vec2