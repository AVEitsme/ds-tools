import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PSI:
    """Represents class to eval Population Stability Index"""
    def __init__(self, actual_sample: pd.DataFrame, expected_sample: pd.DataFrame, bin_counts: int):
        self.actual_sample = actual_sample
        self.expected_sample = expected_sample
        assert all(self.actual_sample.columns == self.expected_sample.columns)
        self.bin_counts = bin_counts
    
    def psi_sum(self) -> float:
        """Sum of all PSI scores in dataset."""
        psi_scores = np.empty(self.actual_sample.columns.shape[0])
        for index, column in enumerate(self.actual_sample.columns):
            psi_scores[index] = self.eval_psi(
                actual=self.actual_sample[column].to_numpy(),
                expected=self.expected_sample[column].to_numpy()
            )
        return psi_scores.sum() / len(psi_scores)

    def eval_psi(self, actual: np.ndarray, expected: np.ndarray) -> float:
        """Eval PSI."""
        actual.sort()
        expected.sort()
        actual_percentage = np.array(np.histogram(actual, self.bin_counts)[0]) / len(actual) + 1
        expected_percentage = np.array(np.histogram(expected, self.bin_counts)[0]) / len(expected) + 1
        return ((actual_percentage - expected_percentage) * np.log(actual_percentage / expected_percentage)).sum()


test_df = pd.read_csv("/home/aveitsme/Projects/ds-tools/psi/test.csv")
t_df, s_df, _, _ = train_test_split(
    test_df.loc[:, test_df.columns != "Transported"],
    test_df["Transported"],
    test_size=.5,
    shuffle=True
)
# s_df += np.random.normal(1, .1, s_df.shape)
print(PSI(t_df, s_df, 10).psi_sum())