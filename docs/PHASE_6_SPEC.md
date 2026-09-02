# AeroTwin-4 Phase 6 Specification: Supervised Fault Diagnosis & Component Identification

## 1. Overview
Phase 6 implements a physics-informed supervised fault-diagnosis system for **AeroTwin-4**.

While Phase 5 answers:
> **"Is the engine anomalous?"**

Phase 6 answers:
> **"Which engine component/fault family is responsible for the anomaly?"**

Phase 6 produces canonical `FaultDiagnosis` records containing predicted fault class (`HEALTHY`, `CYLINDER_1`, `CYLINDER_3`, `BEARING`, `COOLING`, `LUBRICATION`), confidence score, and class probability distribution.

Phase 6 does **NOT** perform severity estimation, RUL prediction, SHAP explanations, or final health scoring.

---

## 2. Target Fault Classes (6 Classes)
0. **`HEALTHY`**: Normal healthy engine operation.
1. **`CYLINDER_1`**: Combustion degradation isolated to Cylinder 1.
2. **`CYLINDER_3`**: Combustion degradation isolated to Cylinder 3.
3. **`BEARING`**: Mechanical bearing friction degradation.
4. **`COOLING`**: Heat rejection capability degradation.
5. **`LUBRICATION`**: Oil pump pressure & viscosity degradation.

---

## 3. Dedicated Severity Run-Level Split
To evaluate component diagnosis on unseen severity levels rather than window memorization:
- **TRAIN**: `HEALTHY_001`, `HEALTHY_002` + `SEV020` & `SEV040` degraded runs across all 5 fault families.
- **VALIDATION**: `HEALTHY_003` (50%) + `SEV060` degraded runs.
- **TEST**: `HEALTHY_003` (50%) + `SEV080` degraded runs (evaluates component diagnosis on unseen severe degradation!).

---

## 4. Strict Leakage Prevention
Ground-truth fields (`gt_degradation_type`, `gt_target_component`, `gt_active_severity`, `gt_current_health`, `gt_is_degraded`), `run_id`, and Phase 5 `anomaly_score` are strictly isolated as evaluation targets and **NEVER** entered into feature matrix $X$.

---

## 5. Four Diagnostic Models Implemented
1. **Model 1 (Physics Rule Baseline)**: Domain-specific residual threshold heuristic rules.
2. **Model 2 (Random Forest)**: Scikit-learn `RandomForestClassifier(class_weight="balanced")`.
3. **Model 3 (Gradient Boosting)**: Scikit-learn `HistGradientBoostingClassifier(class_weight="balanced")`.
4. **Model 4 (PyTorch Supervised MLP)**: 3-layer neural network trained via class-weighted `CrossEntropyLoss`.
