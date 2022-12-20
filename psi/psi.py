import numpy as np
import pandas as pd
from typing import Dict


def scale_range(source: np.ndarray, _min: float, _max: float) -> np.ndarray:
    """Scale range according min & max."""
    source -= source.min()
    source /= source.max() / (_max - _min)
    source += _min
    return source


class PSI:
    """Represents class to eval Population Stability Index."""
    def __init__(self, initial_sample: pd.DataFrame, new_sample: pd.DataFrame):
        assert all(initial_sample.columns == new_sample.columns)
        self.initial_sample = initial_sample
        self.new_sample = new_sample
    
    def eval_psi(self, bin_counts, epsilon) -> Dict[str, float]:
        """Return {column: PSI score} for all features."""
        psi_scores = {}
        for column in self.initial_sample.columns:
            psi_scores[column] = self.psi(
                initial=self.initial_sample[column].to_numpy(),
                new=self.new_sample[column].to_numpy(),
                bin_counts=bin_counts,
                epsilon=epsilon
            ) 
        return psi_scores

    @staticmethod
    def psi(initial: np.ndarray, new: np.ndarray, bin_counts: int, epsilon: float) -> float:
        """Eval PSI for a single feature."""
        initial.sort()
        new.sort()
        bins = scale_range(
            source=np.arange(0, bin_counts + 1, dtype=float),
            _min=min(initial.min(), new.min()),
            _max=max(initial.max(), new.max())
        )
        initial_percentage = np.array(np.histogram(initial, bins)[0]) / len(initial)
        new_percentage = np.array(np.histogram(new, bins)[0]) / len(new)
        initial_percentage[initial_percentage == 0] = epsilon
        new_percentage[new_percentage == 0] = epsilon
        return ((initial_percentage - new_percentage) * (np.log(initial_percentage) - np.log(new_percentage))).sum()
