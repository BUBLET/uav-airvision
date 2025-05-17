class FeatureMetaData(object):
    """
    Содержит необходимую информацию о признаке для удобного доступа.
    """
    def __init__(self):
        self.id = None           # int
        self.response = None     # float
        self.lifetime = None     # int
        self.cam0_point = None   # vec2
        self.cam1_point = None   # vec2