"""
AeroTwin-4 Phase 6 Model Trainer CLI.

Trains Physics Rule Baseline, Random Forest, HistGradientBoosting, and PyTorch MLP
across RAW, RESIDUAL, and HYBRID feature configurations.
Fits FeatureScaler strictly on Training features.
Saves model artifacts under models/phase6/.
"""

import os
import sys
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from AeroTwin.ml.diagnosis.preprocessing import FeatureScaler
from AeroTwin.ml.diagnosis.baselines import RuleBaselineClassifier
from AeroTwin.ml.diagnosis.random_forest import RandomForestDiagnosisModel
from AeroTwin.ml.diagnosis.gradient_boosting import GradientBoostingDiagnosisModel
from AeroTwin.ml.diagnosis.neural_network import PyTorchFaultClassifier


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 6 Fault Diagnosis Model Trainer")
    print("=" * 60)

    configs = ["RAW", "RESIDUAL", "HYBRID"]

    for cfg in configs:
        print(f"\n==================== Training Models for Config: {cfg} ====================")
        feat_dir = os.path.join(_root_dir, "data", "generated", "phase6", "features", cfg.lower())

        X_tr = pd.read_csv(os.path.join(feat_dir, "train", "X_train.csv"))
        Y_tr = pd.read_csv(os.path.join(feat_dir, "train", "Y_train.csv"))
        X_val = pd.read_csv(os.path.join(feat_dir, "val", "X_val.csv"))
        Y_val = pd.read_csv(os.path.join(feat_dir, "val", "Y_val.csv"))

        y_tr = Y_tr["target_fault_idx"].values.astype(int)
        y_val = Y_val["target_fault_idx"].values.astype(int)

        # 1. Fit Feature Scaler STRICTLY on Training data
        scaler_dir = os.path.join(_root_dir, "models", "phase6", cfg.lower(), "preprocessing")
        scaler = FeatureScaler().fit(X_tr)
        scaler.save(os.path.join(scaler_dir, "scaler.json"))

        X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=scaler.feature_names)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=scaler.feature_names)

        # 2. Train Model 1: Physics Rule Baseline
        print(f"  [Model 1: Physics Rule Baseline] Initializing ({cfg})...")
        rule_model = RuleBaselineClassifier().fit(X_tr_s, y_tr)

        # 3. Train Model 2: Random Forest
        print(f"  [Model 2: Random Forest] Training ({cfg})...")
        rf_model = RandomForestDiagnosisModel(n_estimators=100, max_depth=12, random_state=42)
        rf_model.fit(X_tr_s, y_tr)
        rf_dir = os.path.join(_root_dir, "models", "phase6", cfg.lower(), "random_forest")
        rf_model.save(os.path.join(rf_dir, "random_forest_model.joblib"))

        # 4. Train Model 3: Gradient Boosting (HistGradientBoosting)
        print(f"  [Model 3: HistGradientBoosting] Training ({cfg})...")
        gb_model = GradientBoostingDiagnosisModel(max_iter=100, random_state=42)
        gb_model.fit(X_tr_s, y_tr)
        gb_dir = os.path.join(_root_dir, "models", "phase6", cfg.lower(), "gradient_boosting")
        gb_model.save(os.path.join(gb_dir, "gradient_boosting_model.joblib"))

        # 5. Train Model 4: PyTorch Supervised MLP
        print(f"  [Model 4: PyTorch Supervised MLP] Training ({cfg})...")
        mlp_model = PyTorchFaultClassifier(hidden_dim1=64, hidden_dim2=32, epochs=60, random_seed=42)
        mlp_model.fit(X_tr_s, y_tr, X_val_s, y_val)
        mlp_dir = os.path.join(_root_dir, "models", "phase6", cfg.lower(), "neural_network")
        mlp_model.save(mlp_dir)

    print("\n" + "-" * 40)
    print("Phase 6 Model Training & Artifact Saving Complete!")
    print("-" * 40)


if __name__ == "__main__":
    main()
