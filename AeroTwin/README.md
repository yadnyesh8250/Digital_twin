# AeroTwin-4 — Digital Twin for Representative Aero Piston Engine

**AeroTwin-4** is a physics-informed, modular Digital Twin for a representative 4-cylinder 4-stroke aero piston engine developed for the Smart India Hackathon (SIH) project.

---

## 🚦 System Implementation Status

| Phase | Description | Status | Test Coverage |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Engine Mathematical Model & Physics Subsystems | **LOCKED** ✅ | 23 / 23 Tests Passed |
| **Phase 2 & 2.1** | Real-Time Engine Runtime & Canonical Telemetry | **LOCKED** ✅ | 13 / 13 Tests Passed |
| **Phase 3** | Degradation Physics & Ground-Truth Dataset Pipeline | **LOCKED** ✅ | 11 / 11 Tests Passed |
| **Total** | Integrated Test Suite Baseline | **LOCKED** ✅ | **47 / 47 Tests Passed (100%)** |

---

## 🛠️ Implemented Functionalities (Phases 1–3)

### 1. Phase 1 — Engine Mathematical Model & Subsystems
- **Crankshaft Rotational Dynamics**: Rotational dynamics differential equation:
  $$J \frac{d\omega}{dt} = T_{\text{engine}} - T_{\text{load}} - T_{\text{friction}}$$
  where $J = 0.20\text{ kg}\cdot\text{m}^2$, with physically stable equilibrium balance (~3,300 RPM max).
- **RPM-Dependent Engine Torque Capability**: Normalized piston engine torque curve:
  $$\eta(\text{RPM}) = \max\left(0.10, 1.0 - 0.8 \cdot \left(\frac{\text{RPM} - 2500}{3500}\right)^2\right)$$
- **4-Cylinder 4-Stroke Combustion Model**: $720^\circ$ crank angle cycle tracking, firing order **1-3-4-2**, cylinder phase offsets ($0^\circ, 180^\circ, 360^\circ, 540^\circ$), exact cycle-mean torque conservation.
- **Thermal Subsystem**: Lumped energy-balance differential equations ($C \frac{dT}{dt} = Q_{\text{gen}} - Q_{\text{cool}}$) for Cylinder Head Temperature (CHT), Exhaust Gas Temperature (EGT), and Oil Temperature.
- **Lubrication Subsystem**: Engine-driven oil pump, pressure relief valve, oil temperature viscosity friction factor coupling.
- **Fuel Subsystem**: Brake Specific Fuel Consumption (BSFC) mass flow rate ($\text{kg/s}$ & $\text{L/h}$) and regulated fuel pressure.
- **Mechanical Vibration Subsystem**: High-frequency RMS vibration derived from instantaneous torque fluctuation $|T_{\text{instant}} - T_{\text{mean}}|$ and rotational speed imbalance.

---

### 2. Phase 2 & 2.1 — Real-Time Engine Runtime & Telemetry System
- **Independent Simulation Clock (`SimulationClock`)**: Decouples simulation time $t$, step count, and step size $dt$ ($10\text{ms}$ / $100\text{ Hz}$).
- **Engine Runner (`EngineRunner`)**: State machine managing lifecycle states (`STOPPED`, `RUNNING`, `PAUSED`), deterministic random seeds (`seed=42`), dynamic manual input overrides.
- **Dual-Mode Execution Semantics**:
  1. **Real-Time $1\times$ Playback Mode (`run_realtime`)**: Synchronized wall-clock pacing for live streaming to downstream APIs (WebSocket / React dashboard).
  2. **Fast Batch Computation Mode (`run_for`)**: High-throughput calculation rate (~70 kHz) for offline dataset generation.
- **Operating Modes & Mission Profiles (`FlightProfile`)**: Configurable mission profiles transitioning across `IDLE`, `TAXI`, `TAKEOFF`, `CLIMB`, `CRUISE`, `DESCENT`.
- **Canonical Multi-Channel Telemetry (`EngineTelemetry`)**: Standardized 25-field data contract including:
  - Time & Scenario: `timestamp`, `simulation_time`, `engine_id`, `operating_mode`
  - Motion: `throttle`, `rpm`, `crank_angle`
  - Torques: `mean_torque`, `instant_torque`, `load_torque`, `friction_torque`, `net_torque`
  - **Per-Cylinder Torques**: `cylinder_1_torque`, `cylinder_2_torque`, `cylinder_3_torque`, `cylinder_4_torque`
  - Subsystem Sensors: `cht`, `egt`, `oil_temperature`, `oil_pressure`, `oil_pressure_psi`, `fuel_flow`, `fuel_flow_lph`, `fuel_pressure`, `vibration`
- **Telemetry Exporter (`TelemetryExporter`)**: Automated CSV & Parquet telemetry logging.

---

### 3. Phase 3 — Degradation Physics & Ground-Truth Dataset Pipeline
- **Physics-Injected Degradation Mechanisms**:
  - **D1 Cylinder Combustion Degradation**: `combustion_efficiencies` dictionary $[0.50, 1.00]$ scaling per-cylinder power stroke pulse.
  - **D2 Bearing Mechanical Degradation**: `bearing_friction_multiplier` $\ge 1.00$ modifying crankshaft dry/boundary friction without double-counting fluid viscosity friction.
  - **D3 Cooling System Degradation**: `cooling_efficiency` $[0.50, 1.00]$ scaling cylinder head heat rejection rate $Q_{\text{cool}}$.
  - **D4 Lubrication System Degradation**: `lubrication_efficiency` $[0.50, 1.00]$ modifying oil pump pressure capacity and oil viscosity friction factor internally.
- **Time-Dependent Trajectories (`DegradationTrajectoryCalculator`)**: `CONSTANT`, `LINEAR`, `STEP`, `EXPONENTIAL` health degradation profiles over simulation time.
- **Dual Ground-Truth Granularity**:
  - `RunGroundTruth`: Scenario metadata (`run_id`, `degradation_type`, `target_component`, `max_severity`, `trajectory_type`, `seed`, `operating_profile`).
  - `SampleGroundTruth`: Time-step metadata (`timestamp`, `simulation_time`, `degradation_type`, `target_component`, `active_severity`, `current_health`, `is_degraded`).
  - Ground truth is stored separately and is **never** included in ML sensor input feature sets.
- **Sliding-Window Dataset Pipeline (`DatasetBuilder`)**:
  - 5.0s window size, 1.0s stride default (500 samples/window at 100 Hz).
  - Time-domain feature aggregations (mean, std, min, max, rms, peak).
  - **Run-Based Non-Leakage Partitioning**: Train / Validation / Test sets are split strictly by **simulation run IDs**, preventing window overlap leakage.
- **Dataset CLI & Validation Tools**:
  - [`scripts/generate_phase3_dataset.py`](file:///Users/yadnyesh8250/Desktop/Digital_Twin/scripts/generate_phase3_dataset.py): Pilot (`--pilot`) and full (`--full`) dataset generator.
  - [`scripts/validate_phase3_dataset.py`](file:///Users/yadnyesh8250/Desktop/Digital_Twin/scripts/validate_phase3_dataset.py): Automated numerical integrity & physical causality trend validator.

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
│   │   ├── engine/
│   │   │   ├── parameters.py          # Centralized engine parameters
│   │   │   ├── dynamics.py            # Crankshaft rotational dynamics solver
│   │   │   ├── cylinders.py           # 4-cylinder 720° combustion pulse model
│   │   │   ├── thermal.py             # Lumped CHT/EGT/Oil thermal model
│   │   │   ├── lubrication.py         # Oil pump & pressure/viscosity model
│   │   │   ├── fuel.py                # BSFC fuel flow & pressure model
│   │   │   └── vibration.py           # Torque fluctuation vibration model
│   │   └── tests/                     # 23 Phase 1 unit tests
│   │
│   ├── simulator/                     # PHASE 2: Real-Time Engine Runtime
│   │   ├── clock.py                   # SimulationClock (t, dt)
│   │   ├── runner.py                  # EngineRunner state machine & real-time pacing
│   │   ├── scenarios/                 # Operating modes (IDLE..DESCENT) & FlightProfile
│   │   ├── telemetry/                 # Canonical EngineTelemetry schema & exporter
│   │   ├── benchmark.py               # Throughput latency benchmark utility
│   │   └── tests/                     # 13 Phase 2 unit tests
│   │
│   └── degradation/                   # PHASE 3: Degradation Physics & Dataset
│       ├── config.py                  # DegradationConfig, Severity, Component enums
│       ├── mechanisms.py              # D1-D4 physical parameter mapper
│       ├── trajectory.py              # CONSTANT, LINEAR, STEP, EXPONENTIAL profiles
│       ├── ground_truth.py            # Dual RunGroundTruth & SampleGroundTruth
│       ├── injector.py                # DegradationInjector step-by-step runner wrapper
│       ├── dataset.py                 # SlidingWindowGenerator & DatasetBuilder
│       ├── validators.py              # Physical causality & non-leakage validator
│       └── tests/                     # 11 Phase 3 unit tests
│
├── scripts/
│   ├── generate_phase3_dataset.py     # Pilot & Full dataset generation CLI
│   └── validate_phase3_dataset.py     # Dataset causality & integrity validator CLI
│
├── data/
│   └── generated/
│       ├── telemetry_run.csv          # Phase 2 flight scenario telemetry run
│       └── phase3/
│           ├── pilot/                 # 5 pilot simulation run datasets
│           └── full/                  # 23 full simulation run datasets (raw & windows)
│
└── docs/
    ├── PHASE_3_SPEC.md                # Phase 3 technical specification
    ├── PHASE_3_DATASET.md             # Phase 3 dataset layout & feature spec
    └── PHASE_3_PHYSICS.md             # Phase 3 phenomenological severity mapping spec
```

---

## 🚀 Execution & Verification Commands

### 1. Run Complete Automated Test Suite (47/47 Passed)
```bash
.venv/bin/pytest AeroTwin/phase1/tests/ AeroTwin/simulator/tests/ AeroTwin/degradation/tests/ -v
```

### 2. Run Real-Time Engine Simulator & Benchmark
```bash
.venv/bin/python AeroTwin/simulator/main.py
```

### 3. Generate Deterministic Pilot Dataset
```bash
.venv/bin/python scripts/generate_phase3_dataset.py --pilot
```

### 4. Validate Generated Dataset
```bash
.venv/bin/python scripts/validate_phase3_dataset.py data/generated/phase3/pilot
```

### 5. Generate Full Phase 3 Dataset
```bash
.venv/bin/python scripts/generate_phase3_dataset.py --full
.venv/bin/python scripts/validate_phase3_dataset.py data/generated/phase3/full
```

---

## ⚠️ Engineering Disclaimer

Mappings from degradation severity $S \in [0.0, 1.0]$ to physical parameters are **phenomenological reduced-order prototype simulation assumptions**.
Passing unit tests and causality checks confirms **mathematical and physical self-consistency** within the model equations, but does not constitute empirical validation against classified or proprietary UAV engine test cell data.
