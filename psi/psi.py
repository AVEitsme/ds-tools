import numpy as np
import pandas as pd
from typing import Dict


class PSI:
    """Represents class to eval Population Stability Index"""
    def __init__(self, actual_sample: pd.DataFrame, expected_sample: pd.DataFrame, bin_counts: int):
        self.actual_sample = actual_sample
        self.expected_sample = expected_sample
        assert all(self.actual_sample.columns == self.expected_sample.columns)
        self.bin_counts = bin_counts
    
    def eval_psi(self) -> Dict[str, float]:
        """Return column: PSI score for all features."""
        psi_scores = {}
        for column in self.actual_sample.columns:
            psi_scores[column] =self.psi(
                actual=self.actual_sample[column].to_numpy(),
                expected=self.expected_sample[column].to_numpy()
            ) 
        return psi_scores

    def psi(self, actual: np.ndarray, expected: np.ndarray) -> float:
        """Eval PSI for single feature."""
        actual.sort()
        expected.sort()
        actual_percentage = np.array(np.histogram(actual, self.bin_counts)[0]) / len(actual) + 1
        expected_percentage = np.array(np.histogram(expected, self.bin_counts)[0]) / len(expected) + 1
        return ((actual_percentage - expected_percentage) * np.log(actual_percentage / expected_percentage)).sum()
