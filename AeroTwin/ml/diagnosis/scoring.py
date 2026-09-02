"""
AeroTwin-4 Phase 6 Diagnostic Output Contract & Scorer.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from AeroTwin.ml.diagnosis.labels import FAULT_CLASS_ORDER, decode_fault_label


@dataclass
class FaultDiagnosis:
    """
    Canonical Fault Diagnosis Output Record.
    """

    timestamp: float
    simulation_time: float
    predicted_fault: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    feature_configuration: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiagnosisScorer:
    """
    Common Diagnostic Scorer wrapping trained models.
    """

    def __init__(self, model: Any, model_name: str, feature_configuration: str):
        self.model = model
        self.model_name = model_name
        self.feature_configuration = feature_configuration

    def diagnose_window(self, X_window: pd.DataFrame, timestamp: float = 0.0, sim_time: float = 0.0) -> FaultDiagnosis:
        """
        Produce FaultDiagnosis output record for a single window feature vector.
        """
        probs_mat = self.model.predict_proba(X_window)
        probs_vec = probs_mat[0]

        max_idx = int(np.argmax(probs_vec))
        pred_fault = decode_fault_label(max_idx)
        confidence = float(probs_vec[max_idx])

        prob_dict = {fc.value: float(probs_vec[idx]) for idx, fc in enumerate(FAULT_CLASS_ORDER)}

        return FaultDiagnosis(
            timestamp=timestamp,
            simulation_time=sim_time,
            predicted_fault=pred_fault,
            confidence=confidence,
            probabilities=prob_dict,
            model_name=self.model_name,
            feature_configuration=self.feature_configuration,
        )
