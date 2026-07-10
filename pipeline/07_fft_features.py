import os
import re
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root, results_dir, metadata_dir

try:
    from bombusal.io import load_dataset
    HAS_LOADER = True
except Exception:
    HAS_LOADER = False

from bombusal.fft_features import FFTQualityAnalyzer

# -------------------- Configuration --------------------
DATA_ROOT = str(data_root())
RENDERS_PREFIX = "fix_renders_50_"        # projection folder prefix
PNG_GLOB = "*.png"

# FFT / frequency-domain parameters (tunable)
analyzer = FFTQualityAnalyzer(
    hfr_thrs=(0.15, 0.20, 0.25),
    n_theta=36,
    aniso_rmin=0.08,         # better matches the blob-boundary frequency band
    aniso_rmax=0.35,
    slope_fmin=0.01,
    slope_fmax=0.20,
    # center_crop_keep=0.6,  # uncomment to enable center cropping
)

# Optional: watershed result CSV (fill in the path if available, otherwise leave empty)
# Expected columns: species,sample,region,ws_score or ws_label
WS_CSV = ""  # e.g. "path/to/ws_results.csv"

OUT_SUMMARY_CSV = str(results_dir() / "volume_summaries.csv")
OUT_WITH_WS_CSV = str(results_dir() / "volume_summaries_with_ws.csv")

# Columns used to build the global composite score (linear combination after robust z-score)
GOOD_COLS = [
    "HFR@0.20_median",
    "f80_median",
    "spectral_centroid_median",
]
BAD_COLS = [
    "alpha_1_over_f_median",
    "anisotropy_ratio_p90",
]
STABILITY_COLS = [
    # Some dispersion measures (smaller is better; used as a penalty)
    "HFR@0.20_CV", "anisotropy_ratio_CV", "SNR_like_CV"
]
# ------------------------------------------------

def robust_z(x: np.ndarray) -> np.ndarray:
    """median/MAD z-score (robust standardisation)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * (mad + 1e-12)
    return (x - med) / scale

def compute_global_score(df: pd.DataFrame) -> pd.Series:
    cols_present = [c for c in GOOD_COLS + BAD_COLS + STABILITY_COLS if c in df.columns]
    tmp = df[cols_present].copy()

    # Apply robust z-score to each column
    z = tmp.apply(lambda col: pd.Series(robust_z(col.values), index=col.index))

    # Composite score: good metrics add, bad metrics subtract, stability as penalty
    score = (
        z[[c for c in GOOD_COLS if c in z.columns]].sum(axis=1)
        - z[[c for c in BAD_COLS if c in z.columns]].sum(axis=1)
        - 0.5 * z[[c for c in STABILITY_COLS if c in z.columns]].mean(axis=1)  # average penalty
    )
    return score.rename("CQS_global")

# ------- Global (dataset-wide) risk_global computation (not per-region) -------
def compute_risk_global(df: pd.DataFrame) -> pd.Series:
    """
    Combine dataset-wide robust z-scores into risk_global (larger = higher risk).
    Columns missing from the table are ignored automatically without error.
    """
    def z(col):
        x = df[col].astype(float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        return (x - med) / (1.4826 * (mad + 1e-12))

    pos_feats = [
        "anisotropy_ratio_p90",
        "angle_entropy_CV",
        "anisotropy_ratio_median",
        "angle_entropy_IQR",
        "anisotropy_ratio_CV",
        "anisotropy_ratio_IQR",
    ]
    neg_feats = [
        "angle_entropy_p10",
        "angle_entropy_median",
        "f80_IQR",
        "HFR@0.20_IQR",
        "HFR@0.15_IQR",
    ]
    penalty_feats = [
        "SNR_like_CV",
        "HFR@0.20_CV",
        "anisotropy_ratio_CV",  # this effectively adds another 0.5 weight (1.5 total)
    ]

    score = pd.Series(0.0, index=df.index, name="risk_global")
    for c in pos_feats:
        if c in df.columns:
            score = score + z(c)
    for c in neg_feats:
        if c in df.columns:
            score = score - z(c)
    for c in penalty_feats:
        if c in df.columns:
            score = score + 0.5 * z(c)
    return score

def parse_species_sample_region(renders_dir: Path) -> Tuple[str, str, str]:
    """
    /confocal/<species>/<sample>/fix_renders_50_<region>/
    """
    region = renders_dir.name.replace(RENDERS_PREFIX, "")
    sample = renders_dir.parent.name
    species = renders_dir.parent.parent.name
    return species, sample, region

def find_all_renders_dirs(root: str) -> List[Path]:
    """
    1) Prefer building from the file list returned by load_dataset
    2) Otherwise fallback: walk the tree for subdirs named 'fix_renders_50_*' that contain PNGs
    """
    renders_dirs: List[Path] = []

    if HAS_LOADER:
        files = load_dataset(root)
        seen = set()
        for file_path in files:
            p = Path(file_path)
            # Naming rule used before:
            renders_dir = p.parent / f"{RENDERS_PREFIX}{p.name.split('.')[0]}"
            if renders_dir.is_dir() and any(renders_dir.glob(PNG_GLOB)):
                if renders_dir not in seen:
                    renders_dirs.append(renders_dir)
                    seen.add(renders_dir)

    if not renders_dirs:
        for dirpath, dirnames, _ in os.walk(root):
            for dn in dirnames:
                if dn.startswith(RENDERS_PREFIX):
                    rd = Path(dirpath) / dn
                    if any(rd.glob(PNG_GLOB)):
                        renders_dirs.append(rd)

    renders_dirs = sorted(set(renders_dirs))
    return renders_dirs

def analyze_all(root: str) -> pd.DataFrame:
    rows = []
    renders_dirs = find_all_renders_dirs(root)
    print(f"[INFO] Found {len(renders_dirs)} render folders under {root}")

    for rd in renders_dirs:
        species, sample, region = parse_species_sample_region(rd)
        try:
            per_img_df, summary = analyzer.analyze_folder(str(rd), pattern=PNG_GLOB)
            print(f"[OK] Analyzed {rd}: {len(per_img_df)} images, CompositeScore={summary.get('CompositeScore', np.nan):.3f}")
        except Exception as e:
            print(f"[WARN] Skip {rd}: {e}")
            continue

        # Flatten the summary into a single row and add identifier columns
        row = {"species": species, "sample": sample, "region": region, "renders_dir": str(rd)}
        row.update(summary)
        rows.append(row)

    if not rows:
        raise RuntimeError("No summaries computed. Check directory structure and PNG availability.")
    df = pd.DataFrame(rows)
    return df

def maybe_merge_ws(df: pd.DataFrame, ws_csv: str) -> pd.DataFrame:
    if not ws_csv or not Path(ws_csv).is_file():
        print("[INFO] No watershed CSV provided; skipping merge.")
        return df

    ws = pd.read_csv(ws_csv)
    # Expected columns: species,sample,region,ws_score or ws_label (0/1)
    if "ws_score" not in ws.columns and "ws_label" not in ws.columns:
        raise ValueError("WS CSV must contain 'ws_score' (e.g., IoU) or 'ws_label' (0/1).")

    merged = pd.merge(df, ws, on=["species", "sample", "region"], how="left")
    return merged

def main():
    df = analyze_all(DATA_ROOT)
    df['index_id'] = df['sample'] + '_' + df['region']
    # Compute a globally sortable composite score (cross-volume comparison)
    df["CQS_global"]   = compute_global_score(df)   # kept for reference
    df["risk_global"]  = compute_risk_global(df)    # new risk score

    # Add difference col
    # Load segmentation results
    results_file = results_dir() / "segmentation_analysis_results.csv"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run analyze_segmentation.py first.")

    seg_results = pd.read_csv(results_file)

    # Load manual annotations
    info_file = metadata_dir() / "information.csv"

    manual_data = pd.read_csv(info_file)

    # Merge on species, sample, and al_type
    merged = pd.merge(seg_results, manual_data,
                        on=['species', 'sample', 'al_type'],
                        how='left')
    print(f"Records with manual annotations: {merged['total_glomeruli'].notna().sum()}")
    merged['difference'] = abs(merged['total_instances'] - merged['total_glomeruli'])
    merged['index_id'] = merged['sample'] + '_' + merged['al_type']
    diff_map = merged.set_index('index_id')['difference']
    df['difference'] = df['index_id'].map(diff_map)
    # If a difference column exists, check the correlation in passing (optional)
    if "difference" in df.columns:
        try:
            from scipy.stats import spearmanr, pearsonr
            valid = df[["risk_global","difference"]].dropna()
            if len(valid) >= 3:
                sr = spearmanr(valid["risk_global"], valid["difference"])
                pr = pearsonr(valid["risk_global"], valid["difference"])
                print(f"[INFO] Corr(risk_global, difference): "
                    f"Spearman ρ={sr.statistic:.3f} (p={sr.pvalue:.2e}), "
                    f"Pearson r={pr.statistic:.3f} (p={pr.pvalue:.2e})")
        except Exception:
            pass
    # Save the base summary
    df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"[OK] Saved {OUT_SUMMARY_CSV} ({len(df)} rows)")

    # Merge WS results and run a quick correspondence analysis
    df_ws = maybe_merge_ws(df, WS_CSV)
    if df_ws is not df:
        if "ws_score" in df_ws.columns:
            # Compute Pearson/Spearman correlation
            try:
                from scipy.stats import pearsonr, spearmanr
                valid = df_ws[["CQS_global", "ws_score"]].dropna()
                if len(valid) >= 3:
                    pr = pearsonr(valid["CQS_global"], valid["ws_score"])
                    sr = spearmanr(valid["CQS_global"], valid["ws_score"])
                    print(f"[INFO] Corr(CQS_global, ws_score): Pearson r={pr.statistic:.3f} (p={pr.pvalue:.2e}), "
                          f"Spearman ρ={sr.statistic:.3f} (p={sr.pvalue:.2e})")
            except Exception:
                pass

        if "ws_label" in df_ws.columns:
            # Coarse threshold: use a percentile of CQS_global
            thr = np.nanpercentile(df_ws["CQS_global"], 20)
            pred = (df_ws["CQS_global"] > thr).astype(int)
            df_ws["qa_pred"] = pred
            # If 1 means "good", compare against the label
            if "ws_label" in df_ws.columns and df_ws["ws_label"].notna().any():
                tp = int(((pred == 1) & (df_ws["ws_label"] == 1)).sum())
                tn = int(((pred == 0) & (df_ws["ws_label"] == 0)).sum())
                fp = int(((pred == 1) & (df_ws["ws_label"] == 0)).sum())
                fn = int(((pred == 0) & (df_ws["ws_label"] == 1)).sum())
                print(f"[INFO] Simple 20% threshold confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        df_ws.to_csv(OUT_WITH_WS_CSV, index=False)
        print(f"[OK] Saved {OUT_WITH_WS_CSV} ({len(df_ws)} rows)")

if __name__ == "__main__":
    main()
