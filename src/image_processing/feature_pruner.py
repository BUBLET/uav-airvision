class FeaturePruner:
    def __init__(self, grid_max_feature_num):
        """
        grid_max_feature_num: макс. число точек в ячейке
        """
        self.grid_max_feature_num = grid_max_feature_num

    def prune_features(self):
        """
        Удаляет часть признаков из ячейки сетки, если их слишком много,
        чтобы ограничить число признаков в каждой ячейке.
        """
        for i, features in enumerate(self.curr_features):
            # Continue if the number of features in this grid does
            # not exceed the upper bound.
            if len(features) <= self.config.grid_max_feature_num:
                continue
            self.curr_features[i] = sorted(features, key=lambda x:x.lifetime, 
                reverse=True)[:self.config.grid_max_feature_num]   