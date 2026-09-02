"""
AeroTwin-4 Phase 6 Feature Extraction Engine for Fault Diagnosis.

Supports 3 feature configurations:
- Config A (RAW): Raw telemetry + operating context.
- Config B (RESIDUAL): Digital Twin counterfactual residuals + indicators.
- Config C (HYBRID): Raw + Residuals + Operating context + Fault-Specific Signatures.

INCLUDES STRICT RUNTIME ZERO-LEAKAGE ASSERTION:
Phase 5 anomaly_score / anomaly_flag, ground-truth metadata, and run_id MUST NEVER enter feature matrix X.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from AeroTwin.ml.diagnosis.labels import normalize_target_string, encode_fault_label

FORBIDDEN_DIAGNOSIS_FIELDS = [
    "anomaly_score",
    "anomaly_flag",
    "ae_score",
    "iforest_score",
    "stat_score",
    "gt_degradation_type",
    "gt_target_component",
    "gt_active_severity",
    "gt_current_health",
    "gt_is_degraded",
    "run_id",
    "timestamp",
    "simulation_time",
    "engine_id",
]

RAW_CHANNELS = [
    "obs_rpm",
    "obs_mean_torque",
    "obs_instant_torque",
    "obs_load_torque",
    "obs_friction_torque",
    "obs_net_torque",
    "obs_cylinder_1_torque",
    "obs_cylinder_2_torque",
    "obs_cylinder_3_torque",
    "obs_cylinder_4_torque",
    "obs_cht",
    "obs_egt",
    "obs_oil_temperature",
    "obs_oil_pressure",
    "obs_fuel_flow",
    "obs_fuel_pressure",
    "obs_vibration",
]

RESIDUAL_CHANNELS = [
    "res_signed_rpm",
    "res_signed_mean_torque",
    "res_signed_instant_torque",
    "res_signed_load_torque",
    "res_signed_friction_torque",
    "res_signed_net_torque",
    "res_signed_cylinder_1_torque",
    "res_signed_cylinder_2_torque",
    "res_signed_cylinder_3_torque",
    "res_signed_cylinder_4_torque",
    "res_signed_cht",
    "res_signed_egt",
    "res_signed_oil_temperature",
    "res_signed_oil_pressure",
    "res_signed_fuel_flow",
    "res_signed_fuel_pressure",
    "res_signed_vibration",
    "res_norm_rpm",
    "res_norm_cht",
    "res_norm_egt",
    "res_norm_oil_pressure",
    "res_norm_vibration",
]

INDICATOR_CHANNELS = [
    "ind_thermal_dev",
    "ind_oil_dev",
    "ind_vibration_dev",
    "ind_torque_dev",
]


class FeatureExtractor:
    """
    Supervised Fault Diagnosis Feature Extractor.
    """

    def __init__(self, config_type: str = "HYBRID"):
        self.config_type = config_type.upper()
        if self.config_type not in ["RAW", "RESIDUAL", "HYBRID"]:
            raise ValueError(f"Unknown feature config_type: {config_type}")

    def extract_window_features(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """
        Extract time-domain summary statistics + fault-specific signatures for a single 5.0s window.
        """
        feats: Dict[str, float] = {}

        # Operating Context Features
        feats["ctx_throttle"] = float(df_window["throttle"].mean()) if "throttle" in df_window.columns else 0.5
        feats["ctx_ambient_temp"] = float(df_window["ambient_temperature"].mean()) if "ambient_temperature" in df_window.columns else 25.0

        # Select channels based on configuration
        channels_to_process = []
        if self.config_type in ["RAW", "HYBRID"]:
            channels_to_process.extend([c for c in RAW_CHANNELS if c in df_window.columns])
            # Handle un-prefixed raw channel names if present
            unprefixed_raw = [c[4:] for c in RAW_CHANNELS if c.startswith("obs_") and c[4:] in df_window.columns]
            channels_to_process.extend(unprefixed_raw)

        if self.config_type in ["RESIDUAL", "HYBRID"]:
            channels_to_process.extend([c for c in RESIDUAL_CHANNELS if c in df_window.columns])
            channels_to_process.extend([c for c in INDICATOR_CHANNELS if c in df_window.columns])

        for ch in channels_to_process:
            vals = df_window[ch].values.astype(np.float64)
            prefix = "raw_" if ch in RAW_CHANNELS or not ch.startswith("res_") else "res_"
            if ch in INDICATOR_CHANNELS:
                prefix = "ind_"

            feats[f"{prefix}{ch}_mean"] = float(np.mean(vals))
            feats[f"{prefix}{ch}_std"] = float(np.std(vals))
            feats[f"{prefix}{ch}_min"] = float(np.min(vals))
            feats[f"{prefix}{ch}_max"] = float(np.max(vals))
            feats[f"{prefix}{ch}_rms"] = float(np.sqrt(np.mean(vals ** 2)))
            feats[f"{prefix}{ch}_peak"] = float(np.max(np.abs(vals)))
            feats[f"{prefix}{ch}_p2p"] = float(np.ptp(vals))

            # Linear slope over window
            if len(vals) > 1:
                t = np.arange(len(vals), dtype=np.float64)
                slope, _ = np.polyfit(t, vals, 1)
                feats[f"{prefix}{ch}_slope"] = float(slope)
            else:
                feats[f"{prefix}{ch}_slope"] = 0.0

        # Physics Fault-Specific Signatures (Cylinder imbalance, Thermal, Lubrication, Bearing)
        c1_res = df_window["res_signed_cylinder_1_torque"].mean() if "res_signed_cylinder_1_torque" in df_window.columns else 0.0
        c2_res = df_window["res_signed_cylinder_2_torque"].mean() if "res_signed_cylinder_2_torque" in df_window.columns else 0.0
        c3_res = df_window["res_signed_cylinder_3_torque"].mean() if "res_signed_cylinder_3_torque" in df_window.columns else 0.0
        c4_res = df_window["res_signed_cylinder_4_torque"].mean() if "res_signed_cylinder_4_torque" in df_window.columns else 0.0

        cyl_residuals = [c1_res, c2_res, c3_res, c4_res]
        feats["cyl_res_min"] = float(np.min(cyl_residuals))
        feats["cyl_res_max"] = float(np.max(cyl_residuals))
        feats["cyl_res_range"] = float(np.ptp(cyl_residuals))
        feats["cyl_res_std"] = float(np.std(cyl_residuals))
        feats["cyl_balance_dev_ratio"] = float((np.max(cyl_residuals) - np.min(cyl_residuals)) / (np.abs(np.mean(cyl_residuals)) + 1e-3))

        # Enforce Zero Leakage Assertion
        for forbidden in FORBIDDEN_DIAGNOSIS_FIELDS:
            if forbidden in feats:
                del feats[forbidden]

        return feats

    def extract_dataset(
        self,
        df_telemetry: pd.DataFrame,
        window_size_sec: float = 5.0,
        stride_sec: float = 1.0,
        dt: float = 0.01,
        run_id_override: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract feature matrix X and ground-truth metadata Y across sliding windows.
        """
        window_size = int(round(window_size_sec / dt))
        stride = int(round(stride_sec / dt))

        feature_rows = []
        meta_rows = []

        total_samples = len(df_telemetry)
        if total_samples < window_size:
            return pd.DataFrame(), pd.DataFrame()

        for start in range(0, total_samples - window_size + 1, stride):
            df_win = df_telemetry.iloc[start : start + window_size]
            feats = self.extract_window_features(df_win)

            # Metadata extraction
            gt_type = df_win["gt_degradation_type"].iloc[0] if "gt_degradation_type" in df_win.columns else "HEALTHY"
            gt_comp = df_win["gt_target_component"].iloc[0] if "gt_target_component" in df_win.columns else "NONE"
            gt_sev = float(df_win["gt_active_severity"].max()) if "gt_active_severity" in df_win.columns else 0.0
            
            if run_id_override:
                run_id = run_id_override
            elif "run_id" in df_win.columns:
                run_id = df_win["run_id"].iloc[0]
            else:
                run_id = "UNKNOWN"

            t_start = float(df_win["simulation_time"].iloc[0]) if "simulation_time" in df_win.columns else float(df_win["timestamp"].iloc[0])

            target_class_str = normalize_target_string(gt_type, gt_comp)
            target_class_idx = encode_fault_label(target_class_str)

            meta = {
                "run_id": run_id,
                "simulation_start": t_start,
                "gt_degradation_type": gt_type,
                "gt_target_component": gt_comp,
                "gt_active_severity": gt_sev,
                "target_fault_class": target_class_str,
                "target_fault_idx": target_class_idx,
            }

            feature_rows.append(feats)
            meta_rows.append(meta)

        df_X = pd.DataFrame(feature_rows)
        df_Y = pd.DataFrame(meta_rows)

        # STRICT ZERO LEAKAGE ASSERTION
        for col in df_X.columns:
            for forbidden in FORBIDDEN_DIAGNOSIS_FIELDS:
                assert forbidden not in col, f"CRITICAL LEAKAGE: Forbidden field '{forbidden}' found in feature matrix column '{col}'!"

        return df_X, df_Y
