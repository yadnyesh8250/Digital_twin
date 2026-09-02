"""
Unit tests for Severity-Based Run Splitter.
"""

import unittest
import pandas as pd
from AeroTwin.ml.diagnosis.splits import SeverityRunSplitter


class TestSplits(unittest.TestCase):

    def test_severity_run_partitioning(self):
        df_Y = pd.DataFrame({
            "run_id": [
                "HEALTHY_001", "HEALTHY_002", "HEALTHY_003",
                "CYL1_SEV020", "CYL1_SEV040", "CYL1_SEV060", "CYL1_SEV080",
                "BEARING_SEV020", "BEARING_SEV040", "BEARING_SEV060", "BEARING_SEV080",
            ],
            "target_fault_class": [
                "HEALTHY", "HEALTHY", "HEALTHY",
                "CYLINDER_1", "CYLINDER_1", "CYLINDER_1", "CYLINDER_1",
                "BEARING", "BEARING", "BEARING", "BEARING",
            ]
        })
        df_X = pd.DataFrame({"feat1": range(len(df_Y))})

        splitter = SeverityRunSplitter()
        splits = splitter.split_dataset(df_X, df_Y)

        X_train, Y_train = splits["train"]
        X_val, Y_val = splits["val"]
        X_test, Y_test = splits["test"]

        # Train runs must contain SEV020/SEV040
        self.assertTrue(all("SEV020" in r or "SEV040" in r or "HEALTHY_001" in r or "HEALTHY_002" in r for r in Y_train["run_id"]))
        # Val runs must contain SEV060 or HEALTHY_003
        self.assertTrue(all("SEV060" in r or "HEALTHY_003" in r for r in Y_val["run_id"]))
        # Test runs must contain SEV080 or HEALTHY_003 held-out
        self.assertTrue(all("SEV080" in r or "HEALTHY_003" in r for r in Y_test["run_id"]))


if __name__ == "__main__":
    unittest.main()
