class Objective:
    def __init__(self, x: pd.DataFrame | pd.Series, y: pd.Series):
        self.x = x
        self.y = y

    def __call__(self, trial) -> float:
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 0.125),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "subsample": trial.suggest_float("subsample", .5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, 1)
        }
        model = LGBMClassifier(**search_space)
        cross_valid_scores = cross_val_score(model, self.x, self.y, cv=5, scoring="f1")

        return cross_valid_scores.mean()