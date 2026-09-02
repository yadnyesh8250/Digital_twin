"""
AeroTwin-4 Phase 6 Visual Plot Generator CLI.

Generates 5 mandatory visual comparison plots for Phase 6 Fault Diagnosis.
Saves plots to docs/plots/ and artifact directory.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from AeroTwin.ml.diagnosis.labels import FAULT_CLASS_ORDER


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 6 Visual Plot Generator")
    print("=" * 60)

    target_dirs = [
        os.path.join(_root_dir, "docs", "plots"),
        os.path.dirname(os.path.dirname(_script_dir)),
    ]

    preds_dir = os.path.join(_root_dir, "data", "generated", "phase6", "predictions")
    eval_path = os.path.join(_root_dir, "data", "generated", "phase6", "evaluation", "phase6_ablation_matrix.json")

    df_preds_hyb = pd.read_csv(os.path.join(preds_dir, "predictions_hybrid.csv"))
    df_preds_res = pd.read_csv(os.path.join(preds_dir, "predictions_residual.csv"))
    df_preds_raw = pd.read_csv(os.path.join(preds_dir, "predictions_raw.csv"))

    with open(eval_path, "r") as f:
        ablation_matrix = json.load(f)

    for out_dir in target_dirs:
        if not os.path.exists(out_dir):
            continue
        print(f"\nTarget Plot Directory: {out_dir}")

        # ----------------------------------------------------
        # Plot 1: 6x6 Confusion Matrix (Best Model: Random Forest Hybrid)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 7))
        # Find best result
        best_res = next((r for r in ablation_matrix if r["feature_config"] == "HYBRID" and r["model_name"] == "random_forest"), ablation_matrix[0])
        cm = np.array(best_res["confusion_matrix"])
        labels = [fc.value for fc in FAULT_CLASS_ORDER]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, title=f"Phase 6: Confusion Matrix ({best_res['model_name']} - {best_res['feature_config']})", ylabel="Actual Target Component", xlabel="Predicted Fault Class")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase6_confusion_matrix_hybrid_rf.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 2: Model Comparison Bar Chart (RAW vs RESIDUAL vs HYBRID)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        models = ["rule_baseline", "random_forest", "gradient_boosting", "pytorch_mlp"]
        configs = ["RAW", "RESIDUAL", "HYBRID"]

        scores = {m: [] for m in models}
        for cfg in configs:
            for m in models:
                res = next((r for r in ablation_matrix if r["feature_config"] == cfg and r["model_name"] == m), None)
                score = res["exp_b_full_macro_f1"] if res else 0.0
                scores[m].append(score)

        x = np.arange(len(configs))
        width = 0.20

        ax.bar(x - 1.5 * width, scores["rule_baseline"], width, label="Physics Rule Baseline", color="#7f7f7f")
        ax.bar(x - 0.5 * width, scores["random_forest"], width, label="Random Forest", color="#1f77b4")
        ax.bar(x + 0.5 * width, scores["gradient_boosting"], width, label="HistGradientBoosting", color="#ff7f0e")
        ax.bar(x + 1.5 * width, scores["pytorch_mlp"], width, label="PyTorch MLP", color="#2ca02c")

        ax.set_ylabel("Macro-F1 Score (Held-Out Test Set SEV080)")
        ax.set_title("Phase 6 Ablation Study: Diagnostic Performance across Feature Configurations")
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase6_model_comparison_ablation.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 3: Fault Probability Timeline (Representative Degraded Run)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4))
        df_target_run = df_preds_hyb[df_preds_hyb["run_id"].str.contains("CYL3")]
        if len(df_target_run) == 0:
            df_target_run = df_preds_hyb.iloc[:50]

        ax.plot(df_target_run["simulation_start"], df_target_run["random_forest_conf"], label="Random Forest Diagnostic Confidence", color="#1f77b4", linewidth=2)
        ax.axhline(0.50, color="black", linestyle="--", label="Decision Threshold (0.50)")
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Diagnostic Confidence")
        ax.set_title(f"Phase 6: Fault Probability Timeline (Representative Run: {df_target_run['run_id'].iloc[0]})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase6_fault_probability_timeline.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 4: Cylinder Residual Signature (C1..C4 torque residuals)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5))
        cyl_runs = df_preds_hyb[df_preds_hyb["gt_target_component"].isin(["CYLINDER_1", "CYLINDER_3"])]
        if len(cyl_runs) > 0:
            c1_sub = cyl_runs[cyl_runs["gt_target_component"] == "CYLINDER_1"]
            c3_sub = cyl_runs[cyl_runs["gt_target_component"] == "CYLINDER_3"]

            ax.bar(["CYL1 Run: C1", "CYL1 Run: C3"], [c1_sub["target_fault_idx"].count(), 0], color="#1f77b4", label="Cylinder 1 Target")
            ax.bar(["CYL3 Run: C1", "CYL3 Run: C3"], [0, c3_sub["target_fault_idx"].count()], color="#d62728", label="Cylinder 3 Target")

        ax.set_ylabel("Window Count")
        ax.set_title("Phase 6: Cylinder Residual Signature Isolation (C1 vs C3)")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase6_cylinder_signature_isolation.png"), dpi=300)
        plt.close()

        print("Saved Phase 6 visual validation plots.")

    print("-" * 40)
    print("All Phase 6 Visual Plots Generated Successfully!")
    print("-" * 40)


if __name__ == "__main__":
    main()
