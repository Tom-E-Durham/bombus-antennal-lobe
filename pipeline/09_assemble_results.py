#!/usr/bin/env python3
"""Stage 09 - merge the pipeline outputs into a single results table.

Joins, on a species_sample_region key:

  * segmentation_analysis_results.csv  -> glomeruli count + AL voxel volume
  * convex_volumes.csv                 -> convex-hull AL volume
  * volume_summaries.csv               -> risk_global, difference (frequency-domain)
  * MLP_results.csv                    -> risk_pred
  * information.csv                    -> gender, manual glomeruli count

Output: results/final_results.csv. Run with --validate to compare against an existing final_results.csv.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import results_dir, metadata_dir  # noqa: E402


def _index(df, sample_col):
    return df["species"] + "_" + df[sample_col] + "_" + df["al_type"]


def assemble():
    res = results_dir()
    meta = metadata_dir()

    # Pillar A: glomeruli count + plain AL voxel volume
    seg = pd.read_csv(res / "segmentation_analysis_results.csv")
    seg["index"] = _index(seg, "sample")

    # Pillar C (FFT): risk_global + difference (region == al_type)
    vol = pd.read_csv(res / "volume_summaries.csv").rename(columns={"region": "al_type"})
    vol["index"] = _index(vol, "sample")

    # Manual reference: gender + manual glomeruli count
    info = pd.read_csv(meta / "information.csv")
    info["index"] = _index(info, "sample")

    out = seg.merge(vol[["index", "risk_global", "difference"]], on="index", how="left")
    out = out.merge(info[["index", "gender", "total_glomeruli"]], on="index", how="left")

    out = out.rename(columns={
        "total_glomeruli": "manual_glomeruli",
        "total_instances": "algorithm_glomeruli",
        "real_volume_um3": "algorithm_volume_um3",
    })

    # Pillar C (MLP): risk_pred
    mlp = pd.read_csv(res / "MLP_results.csv").rename(columns={"region": "al_type"})
    if "risk_score_pred" in mlp.columns:
        out = out.merge(
            mlp[["species", "sample", "al_type", "risk_score_pred"]],
            on=["species", "sample", "al_type"], how="left",
        )

    # Convex-hull AL volume (added as an extra column).
    convex_path = res / "convex_volumes.csv"
    if convex_path.exists():
        cvx = pd.read_csv(convex_path).rename(
            columns={"id": "sample", "total_volume": "convex_volume_um3"})
        out = out.merge(
            cvx[["species", "sample", "al_type", "convex_volume_um3"]],
            on=["species", "sample", "al_type"], how="left",
        )

    cols = ["species", "sample", "al_type", "gender", "manual_glomeruli",
            "algorithm_volume_um3", "algorithm_glomeruli", "difference",
            "risk_global", "risk_score_pred", "convex_volume_um3"]
    out = out[[c for c in cols if c in out.columns]]

    out_path = res / "final_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out)} rows, {len(out.columns)} cols)")
    return out


def validate(out):
    """Compare the core columns against the archived final_results.csv, if present."""
    ref_path = results_dir() / "final_results.csv"
    ref = pd.read_csv(ref_path)
    key = ["species", "sample", "al_type"]
    merged = out.merge(ref, on=key, suffixes=("_new", "_ref"))
    ok = True
    for c in ["algorithm_volume_um3", "algorithm_glomeruli", "difference", "risk_global", "risk_score_pred"]:
        a, b = f"{c}_new", f"{c}_ref"
        if a in merged and b in merged:
            diff = (merged[a] - merged[b]).abs().max()
            print(f"  {c:24s} max abs diff = {diff:.6g}")
            ok = ok and (pd.isna(diff) or diff < 1e-6)
    print("VALIDATION:", "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    result = assemble()
    if "--validate" in sys.argv:
        validate(result)
