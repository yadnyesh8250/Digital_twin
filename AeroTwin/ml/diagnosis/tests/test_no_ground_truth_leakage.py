"""
Unit test asserting strict Zero Ground-Truth Leakage into Phase 6 Feature Matrix.
"""

import unittest
import numpy as np
import pandas as pd
from AeroTwin.ml.diagnosis.features import FeatureExtractor, FORBIDDEN_DIAGNOSIS_FIELDS


class TestZeroGroundTruthLeakage(unittest.TestCase):

    def test_no_forbidden_fields_in_feature_matrix(self):
        n = 600
        df_dummy = pd.DataFrame({
            "timestamp": np.linspace(0, 6.0, n),
            "simulation_time": np.linspace(0, 6.0, n),
            "run_id": ["TEST_RUN_001"] * n,
            "throttle": [0.70] * n,
            "ambient_temperature": [25.0] * n,
            "obs_rpm": [2800.0] * n,
            "res_signed_rpm": [10.0] * n,
            "anomaly_score": [0.95] * n,
            "gt_degradation_type": ["CYLINDER"] * n,
            "gt_target_component": ["CYLINDER_3"] * n,
            "gt_active_severity": [0.60] * n,
            "gt_current_health": [0.40] * n,
            "gt_is_degraded": [True] * n,
        })

        ext = FeatureExtractor(config_type="HYBRID")
        df_X, df_Y = ext.extract_dataset(df_dummy, window_size_sec=5.0, stride_sec=1.0)

        self.assertGreater(len(df_X), 0)
        for col in df_X.columns:
            for forbidden in FORBIDDEN_DIAGNOSIS_FIELDS:
                self.assertNotIn(forbidden, col, f"Forbidden field '{forbidden}' found in feature column '{col}'!")


if __name__ == "__main__":
    unittest.main()
