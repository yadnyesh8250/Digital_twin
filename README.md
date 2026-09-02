# AeroTwin-4 — Digital Twin for Representative Aero Piston Engine

**AeroTwin-4** is a physics-informed, modular Digital Twin for a representative 4-cylinder 4-stroke aero piston engine developed for the Smart India Hackathon (SIH) project.

---

## 🚦 System Implementation Status

| Phase | Description | Status | Test Coverage |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Engine Mathematical Model & Physics Subsystems | **LOCKED** ✅ | 23 / 23 Tests Passed |
| **Phase 2 & 2.1** | Real-Time Engine Runtime & Canonical Telemetry | **LOCKED** ✅ | 13 / 13 Tests Passed |
| **Phase 3** | Degradation Physics & Ground-Truth Dataset Pipeline | **LOCKED** ✅ | 11 / 11 Tests Passed |
| **Phase 4** | Digital Twin State Engine & Residual Generation | **LOCKED** ✅ | 12 / 12 Tests Passed |
| **Phase 5 & 5.1** | Physics-Informed Anomaly Detection & Numerical Validation | **LOCKED** ✅ | 11 / 11 Tests Passed |
| **Phase 6** | Supervised Fault Diagnosis & Component Identification | **LOCKED** ✅ | 11 / 11 Tests Passed |
| **Total** | Integrated Test Suite Baseline | **LOCKED** ✅ | **81 / 81 Tests Passed (100%)** |

---

## 🛠️ Implemented Functionalities (Phases 1–6)

### 1. Phase 1 — Engine Mathematical Model & Subsystems
- **Crankshaft Rotational Dynamics**: Rotational dynamics differential equation:
  $$J \frac{d\omega}{dt} = T_{\text{engine}} - T_{\text{load}} - T_{\text{friction}}$$
  where $J = 0.20\text{ kg}\cdot\text{m}^2$, with physically stable equilibrium balance (~3,300 RPM max).
- **RPM-Dependent Engine Torque Capability**: Normalized piston engine torque curve:
  $$\eta(\text{RPM}) = \max\left(0.10, 1.0 - 0.8 \cdot \left(\frac{\text{RPM} - 2500}{3500}\right)^2\right)$$
- **4-Cylinder 4-Stroke Combustion Model**: $720^\circ$ crank angle cycle tracking, firing order **1-3-4-2**, cylinder phase offsets ($0^\circ, 180^\circ, 360^\circ, 540^\circ$), exact cycle-mean torque conservation.
- **Thermal Subsystem**: Lumped energy-balance differential equations ($C \frac{dT}{dt} = Q_{\text{gen}} - Q_{\text{cool}}$) for Cylinder Head Temperature (CHT), Exhaust Gas Temperature (EGT), and Oil Temperature.
- **Lubrication Subsystem**: Engine-driven oil pump, pressure relief valve, oil temperature viscosity friction factor coupling.
- **Fuel Subsystem**: BSFC fuel flow rate ($\text{kg/s}$ & $\text{L/h}$) and regulated fuel pressure.
- **Mechanical Vibration Subsystem**: RMS vibration derived from instantaneous torque fluctuation $|T_{\text{instant}} - T_{\text{mean}}|$ and rotational speed imbalance.

---

### 2. Phase 2 & 2.1 — Real-Time Engine Runtime & Telemetry System
- **Simulation Clock (`SimulationClock`)**: Decouples simulation time $t$, step count, and step size $dt$ ($10\text{ms}$ / $100\text{ Hz}$).
- **Engine Runner (`EngineRunner`)**: State machine managing lifecycle states (`STOPPED`, `RUNNING`, `PAUSED`), deterministic random seeds (`seed=42`), dynamic manual input overrides.
- **Dual-Mode Execution Semantics**: Real-time $1\times$ wall-clock playback and high-throughput batch simulation (~70 kHz).
- **Operating Modes & Mission Profiles (`FlightProfile`)**: Mission profiles across `IDLE`, `TAXI`, `TAKEOFF`, `CLIMB`, `CRUISE`, `DESCENT`.
- **Canonical Telemetry (`EngineTelemetry`)**: Standardized 25-field data contract including per-cylinder torques ($C_1, C_2, C_3, C_4$).

---

### 3. Phase 3 — Degradation Physics & Ground-Truth Dataset Pipeline
- **Physics-Injected Degradation Mechanisms**:
  - D1 Cylinder Combustion Degradation
  - D2 Bearing Mechanical Friction Degradation
  - D3 Cooling System Heat Rejection Degradation
  - D4 Lubrication System Oil Pressure & Viscosity Degradation
- **Time-Dependent Trajectories (`DegradationTrajectoryCalculator`)**: `CONSTANT`, `LINEAR`, `STEP`, `EXPONENTIAL` health degradation profiles.
- **Dual Ground-Truth Granularity**: `RunGroundTruth` (scenario metadata) & `SampleGroundTruth` (timestep metadata).
- **Sliding-Window Dataset Pipeline (`DatasetBuilder`)**: 5.0s window size, 1.0s stride default (500 samples/window at 100 Hz).

---

### 4. Phase 4 — Digital Twin State Engine & Residual Generation
- **Dual-Mode Healthy State Engine (`HealthyStateModel`)**:
  - **Mode A (Synchronized Counterfactual Twin)**: Parallel healthy simulation (`degradation=OFF`) using identical seed, initial state, $dt$, throttle trajectory, and flight profile.
  - **Mode B (Pointwise Reference Predictor)**: Reference model mapping operating inputs to expected healthy outputs.
- **Strict Field Separation**: Metadata fields have zero residuals calculated; physical outputs have signed and normalized residuals.
- **Conditioned Baseline (`HealthyBaselineModel`)**: Binned interpolation across $(\text{operating\_mode}, \text{throttle}, \text{RPM})$ providing channel-level reference scales and condition-aware healthy standard deviations.
- **Residual Engine (`ResidualGenerator`)** & **Residual Indicators (`ResidualIndicatorEngine`)**.

---

### 5. Phase 5 & 5.1 — Physics-Informed Anomaly Detection & Numerical Validation
- **Strict Zero Ground-Truth Feature Leakage**: Ground-truth labels are strictly isolated for evaluation and are **NEVER** passed into ML feature matrices.
- **Unsupervised Healthy-Only Training**: Models and scalers are fitted strictly on Healthy Training runs (`HEALTHY_001`, `HEALTHY_002`).
- **Phase 5.1 Numerical & Scaling Fix (`FeatureScaler`)**: Enforced a `min_std = 1e-2` variance floor and feature clipping $[-20.0, +20.0]$, restoring PyTorch Autoencoder healthy reconstruction MSE loss from $10^{33}$ to **0.0173 – 0.1076**.
- **Unseen Operating Condition Generalization**: Verified model generalization when trained on `CRUISE`/`IDLE` and tested on `TAKEOFF`/`DESCENT` (`test_profile_generalization.py`), proving the model learns **Engine Health** rather than operating profile.
- **3 Unsupervised Anomaly Detectors**: Statistical Baseline, Scikit-Learn Isolation Forest, PyTorch Feed-Forward Autoencoder.

---

### 6. Phase 6 — Supervised Fault Diagnosis & Component Identification
- **Component-Level Fault Diagnosis**: Given an anomalous engine window (gated by Phase 5), Phase 6 identifies the degraded component (`HEALTHY`, `CYLINDER_1`, `CYLINDER_3`, `BEARING`, `COOLING`, `LUBRICATION`).
- **Phase 5 Anomaly Gating**: Phase 5 output acts strictly as a gate (`NORMAL` -> stop; `ANOMALOUS` -> trigger Phase 6). Phase 5 `anomaly_score` is **NEVER** fed into Phase 6 feature matrix $X$.
- **Severity Partitioning (`SEV020/040` -> `SEV060` -> `SEV080`)**: Trained strictly on lower/medium severities (`SEV020`, `SEV040`) and evaluated on unseen `SEV080` test runs.
- **4 Diagnostic Models**: Physics Rule Baseline, Scikit-Learn Random Forest, HistGradientBoosting (with `class_weight="balanced"`), PyTorch Supervised MLP.
- **Canonical Output Contract (`FaultDiagnosis`)**: Returns `predicted_fault`, `confidence`, and class `probabilities`.

---

## 📁 Repository Structure

```text
Digital_Twin/
├── README.md                          # Project documentation
├── AGENTS.md                          # Master technical roadmap & engineering rules
├── requirements.txt                   # Dependency manifest
│
├── AeroTwin/
│   ├── phase1/                        # PHASE 1: Engine Physics Subsystems
│   ├── simulator/                     # PHASE 2: Real-Time Engine Runtime
│   ├── degradation/                   # PHASE 3: Degradation Physics & Dataset
│   ├── health/                        # PHASE 4: Digital Twin State Engine & Residuals
│   ├── ml/
│   │   ├── anomaly/                   # PHASE 5: Physics-Informed Anomaly Detection
│   │   └── diagnosis/                 # PHASE 6: Supervised Fault Diagnosis
│   │       ├── labels.py              # 6 FaultClass Enum & mapping utilities
│   │       ├── features.py            # FeatureExtractor (Raw, Residual, Hybrid + signatures)
│   │       ├── preprocessing.py       # FeatureScaler (fit-on-train-only, min_std, clip)
│   │       ├── splits.py              # SeverityRunSplitter (Train: SEV020/040, Val: 060, Test: 080)
│   │       ├── baselines.py           # Model 1: Physics Rule Baseline
│   │       ├── random_forest.py       # Model 2: RandomForestClassifier
│   │       ├── gradient_boosting.py  # Model 3: HistGradientBoostingClassifier
│   │       ├── neural_network.py      # Model 4: PyTorch Supervised MLP
│   │       ├── scoring.py             # FaultDiagnosis contract & DiagnosisScorer
│   │       ├── evaluation.py          # Evaluator (Confusion matrix 6x6, Macro F1)
│   │       └── tests/                 # 11 Phase 6 unit tests
│
├── scripts/
│   ├── build_phase6_features.py       # Feature matrix builder CLI
│   ├── train_phase6_models.py         # Model trainer CLI
│   ├── evaluate_phase6_models.py      # Scientific ablation evaluator CLI
│   └── plot_phase6_results.py         # 5 visual plot generator CLI
│
├── data/
│   └── generated/
│       ├── phase3/                    # Raw telemetry & window datasets
│       ├── phase4/                    # Derived residual datasets
│       ├── phase5/                    # Phase 5 feature matrices & predictions
│       └── phase6/                    # Phase 6 feature matrices, predictions, & ablation matrix
│
├── models/
│   └── phase6/                        # Scalers, Baseline, RandomForest, GBoost, and MLP artifacts
│
└── docs/
    ├── PHASE_6_SPEC.md                # Phase 6 specification
    ├── PHASE_6_VALIDATION.md          # Phase 6 scientific ablation study & validation report
    └── plots/                         # Visual validation plots
```

---

## 🚀 Execution & Verification Commands

### 1. Run Complete Integrated Test Suite (81/81 Passed)
```bash
.venv/bin/pytest AeroTwin/phase1/tests/ \
                  AeroTwin/simulator/tests/ \
                  AeroTwin/degradation/tests/ \
                  AeroTwin/health/tests/ \
                  AeroTwin/ml/anomaly/tests/ \
                  AeroTwin/ml/diagnosis/tests/ -v
```

### 2. Build Phase 6 Features & Train Diagnostic Models
```bash
.venv/bin/python scripts/build_phase6_features.py
.venv/bin/python scripts/train_phase6_models.py
```

### 3. Run Scientific Ablation Evaluation & Generate Plots
```bash
.venv/bin/python scripts/evaluate_phase6_models.py
.venv/bin/python scripts/plot_phase6_results.py
```

---

## ⚠️ Engineering Disclaimer

Mappings from degradation severity $S \in [0.0, 1.0]$ to physical parameters and calculated residuals represent **phenomenological reduced-order Digital Twin estimations**.
Passing unit tests and causality checks confirms **mathematical and physical self-consistency** within the model equations, but does not constitute empirical validation against classified or proprietary UAV engine test cell data.
