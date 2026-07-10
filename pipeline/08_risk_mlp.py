#!/usr/bin/env python3
"""Stage 08 - risk score.

Default (inference): loads the trained model and scaler from models/ and scores the samples in
volume_summaries.csv, writing results/MLP_results.csv. Deterministic.

    python pipeline/08_risk_mlp.py

Retrain: refits the scaler and MLP from scratch, overwrites the saved artifacts, and scores with the fresh
model. Needed only when the dataset changes.

    python pipeline/08_risk_mlp.py --train

By default only labelled samples (those with a `difference` value) are scored. Pass --all to also score
every rendered sample (e.g. new, unlabelled data).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import results_dir, models_dir  # noqa: E402
from bombusal import risk_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Risk-score inference (default) or retraining (--train).")
    ap.add_argument("--train", action="store_true", help="Refit the model from scratch and overwrite saved artifacts.")
    ap.add_argument("--all", action="store_true", help="Score every rendered sample, not just labelled ones.")
    args = ap.parse_args()

    df = pd.read_csv(results_dir() / "volume_summaries.csv")

    if args.train:
        model, scaler, cfg = risk_model.train(df, models_dir())
    else:
        model, scaler, cfg = risk_model.load_model(models_dir())
        print(f"Loaded trained model from {models_dir()} (inference mode)")

    score_df = df if args.all else df[df["difference"].notna()].copy()
    out = risk_model.predict(score_df, model, scaler, cfg)

    out_path = results_dir() / "MLP_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
