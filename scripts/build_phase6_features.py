"""
AeroTwin-4 Phase 6 Feature Matrix Builder CLI.

Reads Phase 4 derived residual CSV files, extracts 5.0s window feature matrices
for RAW, RESIDUAL, and HYBRID configurations, and exports datasets to data/generated/phase6/features/.
"""

import os
import sys
import glob
import pandas as pd
from typing import Dict, List

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from AeroTwin.ml.diagnosis.features import FeatureExtractor
from AeroTwin.ml.diagnosis.splits import SeverityRunSplitter


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 6 Feature Matrix Builder")
    print("=" * 60)

    p4_dir = os.path.join(_root_dir, "data", "generated", "phase4", "full")
    csv_files = sorted(glob.glob(os.path.join(p4_dir, "*", "*.csv")) + glob.glob(os.path.join(p4_dir, "*.csv")))

    print(f"Found {len(csv_files)} Phase 4 telemetry residual CSV files.")
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No Phase 4 CSV files found in {p4_dir}")

    configs = ["RAW", "RESIDUAL", "HYBRID"]
    splitter = SeverityRunSplitter()

    for cfg in configs:
        print(f"\nExtracting Features for Config: {cfg}...")
        ext = FeatureExtractor(config_type=cfg)

        X_list = []
        Y_list = []

        for fpath in csv_files:
            df_tel = pd.read_csv(fpath)
            run_id = os.path.basename(fpath).replace("_derived_residuals.csv", "")
            X_run, Y_run = ext.extract_dataset(df_tel, window_size_sec=5.0, stride_sec=1.0, run_id_override=run_id)
            if len(X_run) > 0:
                X_list.append(X_run)
                Y_list.append(Y_run)

        df_X_full = pd.concat(X_list, ignore_index=True)
        df_Y_full = pd.concat(Y_list, ignore_index=True)

        splits = splitter.split_dataset(df_X_full, df_Y_full)

        out_dir = os.path.join(_root_dir, "data", "generated", "phase6", "features", cfg.lower())
        os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)

        for sname, (X_part, Y_part) in splits.items():
            X_part.to_csv(os.path.join(out_dir, sname, f"X_{sname}.csv"), index=False)
            Y_part.to_csv(os.path.join(out_dir, sname, f"Y_{sname}.csv"), index=False)
            print(f"  [{cfg} {sname.upper()}] Shape: X={X_part.shape}, Y={Y_part.shape}")

    print("\n" + "-" * 40)
    print("Phase 6 Feature Generation Complete!")
    print("-" * 40)


if __name__ == "__main__":
    main()
