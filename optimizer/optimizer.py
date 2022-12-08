import optuna

from abc import ABC, abstractmethod


class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    pass


class EarlyStopping(ABC):
    def __init__(self, max_iter: int, score: float | int):
        self.max_iter = max_iter
        self.count = 0
        self.score = score

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if self._compare_function(study.best_value):
            self.count = 0
            self.score = study.best_value
            return
        self.count += 1
        if self.count == self.max_iter:
            raise EarlyStoppingExceeded()

    @abstractmethod
    def _compare_function(self, best_value: float) -> bool:
        pass


class MaximizeEarlyStopping(EarlyStopping):
    def __init__(self, max_iter: int):
        super().__init__(max_iter, float("-inf"))
    
    def _compare_function(self, best_value: float) -> bool:
        return best_value > self.score


class MinimizeEarlyStopping(EarlyStopping):
    def __init__(self, max_iter: int):
        super().__init__(max_iter, float("inf"))
    
    def _compare_function(self, best_value: float) -> bool:
        return best_value < self.score
