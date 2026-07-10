"""bombusal — image-analysis pipeline for the bumblebee antennal-lobe study.

Public API:
    seg_process            marker-controlled 3D watershed AL glomeruli segmentation
    ConfocalDataset        index of the confocal dataset (<species>/<sample>/*.tif)
    load_volume            read a TIFF stack as a float32 volume (optionally standardised)
    build_convex_hull ...  AL convex-hull volume helpers
    FFTQualityAnalyzer     frequency-domain quality features for the risk score
"""
from .al_segmentation import seg_process
from .io import ConfocalDataset, load_volume, load_dataset
from .convex_volume import build_point_cloud, build_convex_hull, mesh_to_shell_volume
from .fft_features import FFTQualityAnalyzer
from .risk_model import MLPWeighting, load_model, predict, train

__all__ = [
    "seg_process",
    "ConfocalDataset",
    "load_volume",
    "load_dataset",
    "build_point_cloud",
    "build_convex_hull",
    "mesh_to_shell_volume",
    "FFTQualityAnalyzer",
    "MLPWeighting",
    "load_model",
    "predict",
    "train",
]
