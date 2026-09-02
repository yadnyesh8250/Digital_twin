"""
AeroTwin-4 Phase 6 Diagnostic Evaluator Engine.

Computes 6x6 Confusion Matrix, Accuracy, Macro F1, Weighted F1, and Per-Class Precision/Recall.
Supports:
- Experiment A: Fault Diagnosis (Degraded Only on SEV080)
- Experiment B: Full Diagnostic Classifier (Including Held-Out Healthy)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)
from AeroTwin.ml.diagnosis.labels import FAULT_CLASS_ORDER, FAULT_TO_IDX


class Evaluator:
    """
    Supervised Fault Diagnosis Evaluation Metrics Engine.
    """

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        experiment_name: str = "Experiment_B_Full",
    ) -> Dict[str, Any]:
        """
        Calculate metrics dictionary.
        """
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        prec, rec, f1, supp = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(FAULT_CLASS_ORDER))), zero_division=0
        )

        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(len(FAULT_CLASS_ORDER)))
        ).tolist()

        per_class: Dict[str, Dict[str, float]] = {}
        for idx, fc in enumerate(FAULT_CLASS_ORDER):
            per_class[fc.value] = {
                "precision": float(prec[idx]),
                "recall": float(rec[idx]),
                "f1_score": float(f1[idx]),
                "support": int(supp[idx]),
            }

        return {
            "experiment": experiment_name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": cm,
            "class_labels": [fc.value for fc in FAULT_CLASS_ORDER],
        }
