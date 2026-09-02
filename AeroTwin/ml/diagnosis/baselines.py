"""
AeroTwin-4 Model 1: Physics/Rule-Based Baseline Classifier.

Uses domain-specific physical residual heuristics to establish a transparent baseline.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from AeroTwin.ml.diagnosis.labels import FaultClass, FAULT_TO_IDX, FAULT_CLASS_ORDER


class RuleBaselineClassifier:
    """
    Physics Rule-Based Diagnostic Classifier.
    """

    def __init__(self):
        self.is_fitted = True

    def fit(self, X_train: pd.DataFrame, y_train: Optional[np.ndarray] = None) -> "RuleBaselineClassifier":
        """
        Rule baseline requires no training, but maintains API compatibility.
        """
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using physical residual rules.
        """
        n_samples = len(X)
        n_classes = len(FAULT_CLASS_ORDER)
        probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        for i in range(n_samples):
            row = X.iloc[i] if isinstance(X, pd.DataFrame) else pd.Series(X[i])

            c1_res = row.get("res_res_signed_cylinder_1_torque_mean", 0.0)
            c3_res = row.get("res_res_signed_cylinder_3_torque_mean", 0.0)
            cht_res = row.get("res_res_signed_cht_mean", 0.0)
            oil_res = row.get("res_res_signed_oil_pressure_mean", 0.0)
            fric_res = row.get("res_res_signed_friction_torque_mean", 0.0)
            vib_res = row.get("res_res_signed_vibration_mean", 0.0)

            # Rule heuristic mapping
            if abs(c3_res) > 5.0 and abs(c3_res) > abs(c1_res) + 2.0:
                predicted_class = FaultClass.CYLINDER_3.value
            elif abs(c1_res) > 5.0 and abs(c1_res) > abs(c3_res) + 2.0:
                predicted_class = FaultClass.CYLINDER_1.value
            elif cht_res > 10.0:
                predicted_class = FaultClass.COOLING.value
            elif oil_res < -20.0 or row.get("ind_ind_oil_dev_mean", 0.0) > 0.5:
                predicted_class = FaultClass.LUBRICATION.value
            elif fric_res > 1.5 or vib_res > 0.15 or row.get("ind_ind_vibration_dev_mean", 0.0) > 0.3:
                predicted_class = FaultClass.BEARING.value
            else:
                predicted_class = FaultClass.HEALTHY.value

            pred_idx = FAULT_TO_IDX[predicted_class]
            probs[i, pred_idx] = 0.90
            for c in range(n_classes):
                if c != pred_idx:
                    probs[i, c] = 0.02

        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class indices (0..5).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
