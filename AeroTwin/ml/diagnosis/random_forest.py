"""
AeroTwin-4 Model 2: Random Forest Supervised Fault Diagnosis Model.

Uses Scikit-Learn RandomForestClassifier with balanced class weighting.
"""

from typing import Optional, Dict, Any, Tuple, List
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from AeroTwin.ml.diagnosis.labels import FAULT_CLASS_ORDER


class RandomForestDiagnosisModel:
    """
    Random Forest Supervised Classifier.
    """

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "RandomForestDiagnosisModel":
        """
        Fit Random Forest classifier on training features and class indices.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.array(X_train)

        self.model.fit(X_mat, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities matrix (n_samples, 6).
        """
        if not self.is_fitted:
            raise RuntimeError("RandomForestDiagnosisModel is not fitted!")

        X_mat = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.array(X)
        raw_probs = self.model.predict_proba(X_mat)

        # Align with canonical 6 classes if some classes are missing in training subset
        n_samples = len(X_mat)
        n_classes = len(FAULT_CLASS_ORDER)
        probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        model_classes = getattr(self.model, "classes_", np.arange(n_classes))
        for col_idx, class_label_idx in enumerate(model_classes):
            probs[:, int(class_label_idx)] = raw_probs[:, col_idx]

        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class indices (0..5).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, filepath: str):
        """
        Save model artifact using joblib.
        """
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, filepath)

    def load(self, filepath: str) -> "RandomForestDiagnosisModel":
        """
        Load model artifact using joblib.
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_fitted = data["is_fitted"]
        return self
