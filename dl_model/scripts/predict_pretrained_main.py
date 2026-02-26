from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import platform, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import time

# add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from dl_model.pipeline.pretrained_predictor import (
    PretrainedPredictConfig,
    PretrainedPredictor,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict using a pretrained Keras regressor + pretrained scaler.")
    p.add_argument("--features-file", required=True)
    p.add_argument("--true-score-name", default="dseq_from_true")

    p.add_argument("--mode", type=int, default=1, choices=[1, 3])
    p.add_argument("--remove-correlated-features", action="store_true")
    p.add_argument("--corr-threshold", type=float, default=0.90)

    p.add_argument("--scaler-type-features", default="standard", choices=["standard", "rank", "zscore"])
    p.add_argument("--scaler-type-labels", default="standard", choices=["standard", "rank", "zscore"])

    p.add_argument("--model-path", required=True)
    p.add_argument("--scaler-path", required=True)

    p.add_argument("--out-dir", default="../out")
    p.add_argument("--run-id", default="0")
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    t0 = time.perf_counter()

    args = build_parser().parse_args(argv)

    cfg = PretrainedPredictConfig(
        features_file=args.features_file,
        true_score_name=args.true_score_name,
        mode=args.mode,
        remove_correlated_features=args.remove_correlated_features,
        corr_threshold=args.corr_threshold,
        scaler_type_features=args.scaler_type_features,
        scaler_type_labels=args.scaler_type_labels,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        out_dir=args.out_dir,
        run_id=args.run_id,
        compute_metrics_if_possible=(not args.no_metrics),
        verbose=(not args.quiet),
    )

    out = PretrainedPredictor(cfg).run(custom_objects=None)
    print(json.dumps(out, indent=2))

    t1 = time.perf_counter()
    total_time = t1 - t0
    print(f"\nTOTAL runtime (config + predict): {total_time:.3f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
