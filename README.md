# Automated segmentation and quantification of the bumblebee antennal lobe

Image-analysis pipeline for *Sexual dimorphisms of the olfactory system are conserved across bumblebee
species* (Oppersma et al.). From confocal stacks of bumblebee brains and antennal lobes (ALs) it produces
glomeruli counts, AL volumes, whole-brain volumes, and a frequency-domain image-quality score used to
select reliable samples.

The dataset covers six species, workers and males: *B. terrestris, B. pascuorum, B. pratorum, B. lapidarius,
B. hypnorum, B. lucorum*.

## Requirements

Python 3.8.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements-freeze.txt` pins the full environment. Stage 06 (2D rendering) uses PyVista/VTK off-screen
rendering; on a headless machine run it under `xvfb-run` or install OSMesa.

## Dataset layout

Set `paths.data_root` in `config.yaml` to a directory organised as:

```
<data_root>/<species>/<sample>/
├── AL1.tif  AL2.tif  wholebrain.tif
└── cache_AL1/  cache_AL2/  cache_wb/      # binary masks + parameters.json
```

## Usage

Run the whole pipeline:

```bash
python run_all.py
```

List stages with `python run_all.py --list`; run a subset with `--only` / `--skip` / `--from`. Stages 02
(segmentation) and 06 (rendering) work on the raw volumes; to regenerate the result tables from cached masks
and existing renders, skip them:

```bash
python run_all.py --skip 02 06
```

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 01 | `pipeline/01_extract_parameters.py` | per-sample `parameters.json` |
| 02 | `pipeline/02_segment_al.py` (via `run_al_batch.sh`) | glomeruli instance labels, `AL*_seg_counts.csv` |
| 03 | `pipeline/03_aggregate_al.py` | `results/segmentation_analysis_results.csv` |
| 04 | `pipeline/04_convex_volume.py` | `results/convex_volumes.csv` |
| 05 | `pipeline/05_segment_wholebrain.py` | whole-brain volumes |
| 06 | `pipeline/06_render.py` | `fix_renders_50_<AL>/` projections |
| 07 | `pipeline/07_fft_features.py` | `results/volume_summaries.csv` (features, `risk_global`) |
| 08 | `pipeline/08_risk_mlp.py` | `results/MLP_results.csv` (`risk_pred`) |
| 09 | `pipeline/09_assemble_results.py` | `results/final_results.csv` |
| 10–11 | `pipeline/10_*`, `pipeline/11_*` | per-glomerulus volumes, AL/brain ratios |

### Method notes

- **Glomeruli segmentation** — marker-controlled 3D watershed: standardise → percentile contrast → Gaussian
  denoise → Otsu threshold → distance transform → peak markers → watershed. Per-sample parameters are read
  from `cache_AL*/parameters.json`.
- **AL volume** — the 3D convex hull of the segmented glomeruli, filled and scaled by voxel size, so it
  captures the AL envelope rather than only the discrete glomeruli.
- **Whole-brain volume** — the largest connected component of a thresholded, morphologically-closed mask,
  times voxel size.
- **Risk score** (`risk_pred`) — an MLP over frequency-domain features of the 2D renders, trained to predict
  the discrepancy between manual and automatic glomeruli counts. Stage 08 runs inference from the saved model
  by default; pass `--train` to refit it when the dataset changes.

## Repository layout

```
src/bombusal/   library: io, al_segmentation, convex_volume, fft_features, risk_model
pipeline/       numbered stage scripts
metadata/       information.csv, ConfocalFileParameters.xlsx, render_angles.csv
models/         trained risk-score model + scaler
results/        result tables
config.yaml     paths and parameters
run_all.py      run the full pipeline
```

## Citation

> Oppersma et al. *Sexual dimorphisms of the olfactory system are conserved across bumblebee species.*

## License

Released under the MIT License. See [LICENSE](LICENSE).
