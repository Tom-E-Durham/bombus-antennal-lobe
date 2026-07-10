#!/usr/bin/env python3
"""Run the full pipeline end to end.

Stages run in order. Each stage reads its paths and parameters from config.yaml.

    python run_all.py                 # run every stage
    python run_all.py --list          # list stages and exit
    python run_all.py --only 08 09    # run selected stages
    python run_all.py --skip 02 06    # run all but the listed stages
    python run_all.py --from 07       # run from a stage to the end

Stages 02 (AL segmentation) and 06 (rendering) are the expensive ones and operate on the raw image
volumes. To regenerate only the result tables from the cached masks and existing renders, skip them:

    python run_all.py --skip 02 06
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("01", "Extract per-sample segmentation parameters", ["python", "pipeline/01_extract_parameters.py"]),
    ("02", "Segment antennal-lobe glomeruli (batch)",     ["bash",   "pipeline/run_al_batch.sh"]),
    ("03", "Aggregate glomeruli counts and AL volumes",   ["python", "pipeline/03_aggregate_al.py"]),
    ("04", "Antennal-lobe convex-hull volumes",           ["python", "pipeline/04_convex_volume.py"]),
    ("05", "Segment whole brain and compute volumes",     ["python", "pipeline/05_segment_wholebrain.py"]),
    ("06", "Render 2D projections",                       ["python", "pipeline/06_render.py"]),
    ("07", "Frequency-domain quality features",           ["python", "pipeline/07_fft_features.py"]),
    ("08", "Predict image-quality risk score",            ["python", "pipeline/08_risk_mlp.py"]),
    ("09", "Merge results into one table",                ["python", "pipeline/09_assemble_results.py"]),
    ("10", "Per-glomerulus volumes",                      ["python", "pipeline/10_glomeruli_volumes.py"]),
    ("11", "AL-to-brain relative volumes",                ["python", "pipeline/11_brain_ratios.py"]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List stages and exit.")
    p.add_argument("--only", nargs="+", metavar="ID", help="Run only these stage IDs.")
    p.add_argument("--skip", nargs="+", metavar="ID", help="Skip these stage IDs.")
    p.add_argument("--from", dest="from_id", metavar="ID", help="Start from this stage ID.")
    p.add_argument("--keep-going", action="store_true", help="Continue after a failing stage.")
    return p.parse_args()


def select(args):
    stages = STAGES
    if args.from_id:
        ids = [s[0] for s in stages]
        if args.from_id not in ids:
            sys.exit(f"Unknown stage '{args.from_id}'")
        stages = stages[ids.index(args.from_id):]
    if args.only:
        stages = [s for s in stages if s[0] in set(args.only)]
    if args.skip:
        stages = [s for s in stages if s[0] not in set(args.skip)]
    return stages


def main():
    args = parse_args()
    if args.list:
        for sid, desc, _ in STAGES:
            print(f"  {sid}  {desc}")
        return

    stages = select(args)
    print(f"Running {len(stages)} stage(s): {', '.join(s[0] for s in stages)}\n")
    for sid, desc, cmd in stages:
        print(f"{'=' * 70}\n[{sid}] {desc}\n{'=' * 70}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"[{sid}] failed (exit {result.returncode})")
            if not args.keep_going:
                sys.exit(result.returncode)
    print("\nDone.")


if __name__ == "__main__":
    main()
