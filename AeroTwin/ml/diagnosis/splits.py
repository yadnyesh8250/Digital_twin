"""
AeroTwin-4 Severity-Based Run Splitter for Supervised Fault Diagnosis.

Partitioning Strategy:
- TRAIN: HEALTHY_001, HEALTHY_002 + SEV020 & SEV040 degraded runs across all 5 fault families.
- VALIDATION: HEALTHY_003 + SEV060 degraded runs.
- TEST: Held-out HEALTHY data + SEV080 degraded runs (evaluates component diagnosis on unseen severity!).
"""

from typing import Tuple, Dict, Any, List
import pandas as pd


class SeverityRunSplitter:
    """
    Run-based partitioner enforcing severity generalization splitting.
    """

    TRAIN_RUN_SUBSTRINGS = ["HEALTHY_001", "HEALTHY_002", "SEV020", "SEV040"]
    VAL_RUN_SUBSTRINGS = ["HEALTHY_003", "SEV060"]
    TEST_RUN_SUBSTRINGS = ["SEV080"]

    def split_dataset(
        self, df_X: pd.DataFrame, df_Y: pd.DataFrame
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split feature matrix X and metadata Y into Train, Validation, and Test partitions.
        """
        if len(df_X) != len(df_Y):
            raise ValueError(f"X and Y length mismatch: {len(df_X)} vs {len(df_Y)}")

        run_ids = df_Y["run_id"].astype(str)

        train_mask = run_ids.apply(lambda r: any(sub in r for sub in self.TRAIN_RUN_SUBSTRINGS))
        val_mask = run_ids.apply(lambda r: any(sub in r for sub in self.VAL_RUN_SUBSTRINGS))
        test_mask = run_ids.apply(lambda r: any(sub in r for sub in self.TEST_RUN_SUBSTRINGS))

        # Include 50% of HEALTHY_003 in Test set to evaluate Experiment B (Full Diagnostic Classifier with Healthy)
        h3_mask = run_ids.str.contains("HEALTHY_003")
        h3_indices = df_Y[h3_mask].index
        if len(h3_indices) > 0:
            half = len(h3_indices) // 2
            h3_val_idx = h3_indices[:half]
            h3_test_idx = h3_indices[half:]

            val_mask = val_mask & (~h3_mask)
            val_mask.loc[h3_val_idx] = True

            test_mask.loc[h3_test_idx] = True

        X_train, Y_train = df_X[train_mask].copy(), df_Y[train_mask].copy()
        X_val, Y_val = df_X[val_mask].copy(), df_Y[val_mask].copy()
        X_test, Y_test = df_X[test_mask].copy(), df_Y[test_mask].copy()

        return {
            "train": (X_train, Y_train),
            "val": (X_val, Y_val),
            "test": (X_test, Y_test),
        }
