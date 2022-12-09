import optuna

from abc import ABC, abstractmethod, ABCMeta
from typing import Callable

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    """Represents class raised to stop optuna study."""
    pass


class EarlyStopping(ABC):
    """Early stopping callback base class."""
    def __init__(self, max_iter: int, score: float | int):
        self.max_iter = max_iter
        self.count = 0
        self.score = score

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback, raise exception if there is no new score in the last N iterations."""
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
    """Represents maximize callback."""
    def __init__(self, max_iter: int):
        super().__init__(max_iter, float("-inf"))
    
    def _compare_function(self, best_value: float) -> bool:
        return best_value > self.score


class MinimizeEarlyStopping(EarlyStopping):
    """Represents minimize callback."""
    def __init__(self, max_iter: int):
        super().__init__(max_iter, float("inf"))
    
    def _compare_function(self, best_value: float) -> bool:
        return best_value < self.score


class OptunaOptimizer(ABC):
    """Represents OptunaOptimizer base class to tune hyperparameters."""
    def __init__(self, objective: Callable, study_name: str):
        self.objective = objective
        self.study_name = study_name

    def optimize(self, n_trials: int, max_iter: int) -> optuna.trial.FrozenTrial:
        """Tune hyperparameters by optimization of objective function."""
        study = optuna.create_study(direction=self.get_direction(), study_name=self.study_name)
        func = lambda trial: self.objective(trial)
        try: 
            study.optimize(
                func=func,
                n_trials=n_trials,
                callbacks=[self.get_callback()(max_iter)],
                show_progress_bar=True
            )
        except EarlyStoppingExceeded:
            print(f"EarlyStopping Exceeded: No new best scores on iters {max_iter}")
        return study.best_trial
    
    @abstractmethod
    def get_direction(self) -> str:
        """Get direction"""

    @abstractmethod
    def get_callback(self) -> ABCMeta:
        """Get callback""" 

class OptunaMaximizer(OptunaOptimizer):
    """Represents optimizer which is maximizes objective function."""
    def get_direction(self) -> str:
        return "maximize"

    def get_callback(self) -> ABCMeta:
        return MaximizeEarlyStopping

class OptunaMinimizer(OptunaOptimizer):
    """Represents optimizer which is minimizes objective function."""
    def get_direction(self) -> str:
        return "minimize"

    def get_callback(self) -> ABCMeta:
        return MinimizeEarlyStopping