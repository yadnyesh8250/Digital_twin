"""
AeroTwin-4 Phase 6 Scientific Ablation Evaluator CLI.

Evaluates Rule Baseline, Random Forest, HistGradientBoosting, and PyTorch MLP
across RAW, RESIDUAL, and HYBRID feature configurations on the held-out Test set.

Evaluates two experimental tracks:
- Experiment A: Fault Diagnosis (Degraded Only on SEV080)
- Experiment B: Full Diagnostic Classifier (Including Held-Out Healthy)
"""

import os
import sys
import json
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
from AeroTwin.ml.diagnosis.evaluation import Evaluator


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 6 Scientific Ablation Evaluator")
    print("=" * 60)

    configs = ["RAW", "RESIDUAL", "HYBRID"]
    evaluator = Evaluator()
    ablation_results = []

    out_eval_dir = os.path.join(_root_dir, "data", "generated", "phase6", "evaluation")
    out_preds_dir = os.path.join(_root_dir, "data", "generated", "phase6", "predictions")
    os.makedirs(out_eval_dir, exist_ok=True)
    os.makedirs(out_preds_dir, exist_ok=True)

    for cfg in configs:
        print(f"\n==================== Evaluating Config: {cfg} ====================")
        feat_dir = os.path.join(_root_dir, "data", "generated", "phase6", "features", cfg.lower())

        X_te = pd.read_csv(os.path.join(feat_dir, "test", "X_test.csv"))
        Y_te = pd.read_csv(os.path.join(feat_dir, "test", "Y_test.csv"))
        y_te_full = Y_te["target_fault_idx"].values.astype(int)

        scaler = FeatureScaler().load(os.path.join(_root_dir, "models", "phase6", cfg.lower(), "preprocessing", "scaler.json"))
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=scaler.feature_names)

        # Degraded Only mask for Experiment A
        deg_mask = Y_te["target_fault_class"] != "HEALTHY"
        X_te_deg = X_te_s[deg_mask]
        y_te_deg = y_te_full[deg_mask]

        # Load models
        rule_model = RuleBaselineClassifier()
        rf_model = RandomForestDiagnosisModel().load(os.path.join(_root_dir, "models", "phase6", cfg.lower(), "random_forest", "random_forest_model.joblib"))
        gb_model = GradientBoostingDiagnosisModel().load(os.path.join(_root_dir, "models", "phase6", cfg.lower(), "gradient_boosting", "gradient_boosting_model.joblib"))
        mlp_model = PyTorchFaultClassifier().load(os.path.join(_root_dir, "models", "phase6", cfg.lower(), "neural_network"))

        models_dict = {
            "rule_baseline": rule_model,
            "random_forest": rf_model,
            "gradient_boosting": gb_model,
            "pytorch_mlp": mlp_model,
        }

        cfg_predictions = Y_te.copy()

        for mname, model in models_dict.items():
            # Experiment A: Degraded Only (SEV080)
            preds_deg = model.predict(X_te_deg)
            metrics_exp_a = evaluator.evaluate_predictions(y_te_deg, preds_deg, experiment_name="Experiment_A_Degraded")

            # Experiment B: Full Classifier (Including Healthy)
            preds_full = model.predict(X_te_s)
            probs_full = model.predict_proba(X_te_s)
            metrics_exp_b = evaluator.evaluate_predictions(y_te_full, preds_full, experiment_name="Experiment_B_Full")

            cfg_predictions[f"{mname}_pred"] = preds_full
            cfg_predictions[f"{mname}_conf"] = np.max(probs_full, axis=1)

            print(f"  [{mname:<18}] Exp A (Degraded) Macro-F1: {metrics_exp_a['macro_f1']:.4f} | Exp B (Full) Macro-F1: {metrics_exp_b['macro_f1']:.4f} | Acc: {metrics_exp_b['accuracy']:.4f}")

            ablation_results.append({
                "feature_config": cfg,
                "model_name": mname,
                "exp_a_degraded_macro_f1": metrics_exp_a["macro_f1"],
                "exp_a_degraded_accuracy": metrics_exp_a["accuracy"],
                "exp_b_full_macro_f1": metrics_exp_b["macro_f1"],
                "exp_b_full_accuracy": metrics_exp_b["accuracy"],
                "exp_b_full_weighted_f1": metrics_exp_b["weighted_f1"],
                "per_class": metrics_exp_b["per_class"],
                "confusion_matrix": metrics_exp_b["confusion_matrix"],
            })

        cfg_predictions.to_csv(os.path.join(out_preds_dir, f"predictions_{cfg.lower()}.csv"), index=False)

    # Save complete ablation matrix JSON
    ablation_json_path = os.path.join(out_eval_dir, "phase6_ablation_matrix.json")
    with open(ablation_json_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 6 ABLATION STUDY COMPARISON MATRIX (HELD-OUT TEST SET SEV080)")
    print("=" * 80)
    print(f"{'Config':<10} | {'Model':<18} | {'Exp A Macro-F1':<15} | {'Exp B Macro-F1':<15} | {'Exp B Accuracy':<15}")
    print("-" * 80)
    for res in ablation_results:
        print(f"{res['feature_config']:<10} | {res['model_name']:<18} | {res['exp_a_degraded_macro_f1']:<15.4f} | {res['exp_b_full_macro_f1']:<15.4f} | {res['exp_b_full_accuracy']:<15.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
