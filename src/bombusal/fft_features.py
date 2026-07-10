import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class FFTQualityAnalyzer:
    """
    Read all PNG images (grayscale) in a folder, compute frequency-domain
    quality metrics, and perform robust aggregation.
    Usage:
        analyzer = FFTQualityAnalyzer(hfr_thrs=(0.15, 0.20, 0.25))
        per_img_df, summary = analyzer.analyze_folder("/path/to/projections", save_csv="metrics.csv")
    """
    EPS = 1e-12

    def __init__(
        self,
        hfr_thrs: Tuple[float, ...] = (0.25,),
        n_theta: int = 36,
        aniso_rmin: float = 0.05,
        aniso_rmax: float = 0.45,
        slope_fmin: float = 0.02,
        slope_fmax: float = 0.45,
    ):
        self.hfr_thrs = hfr_thrs
        self.n_theta = n_theta
        self.aniso_rmin = aniso_rmin
        self.aniso_rmax = aniso_rmax
        self.slope_fmin = slope_fmin
        self.slope_fmax = slope_fmax

        # "Good"/"bad" dimensions used when combining the composite score
        self.GOOD_KEYS_BASE = ["spectral_centroid", "f80", "SNR_like"]
        self.BAD_KEYS = ["alpha_1_over_f", "anisotropy_ratio", "noise_floor"]

    # ---------------------- Single-image core computation ---------------------- #
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img = img - img.mean()
        h, w = img.shape
        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        return img * win

    def _power_spectrum(self, x: np.ndarray) -> np.ndarray:
        F = np.fft.fft2(x)
        F = np.fft.fftshift(F)
        return (np.abs(F) ** 2)

    def _radial_stats(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = P.shape
        cy, cx = (h-1)/2.0, (w-1)/2.0
        yy, xx = np.indices(P.shape)
        ry = (yy - cy) / h
        rx = (xx - cx) / w
        r = np.sqrt(rx*rx + ry*ry)
        rmax = 0.5
        mask = r <= rmax
        r = r[mask]
        v = P[mask]
        nbins = max(8, min(h, w)//2)
        bins = np.linspace(0, rmax, num=nbins)
        idx = np.digitize(r, bins) - 1
        sums = np.bincount(idx, weights=v, minlength=nbins)
        cnts = np.bincount(idx, minlength=nbins).astype(np.float32)
        Pr = sums / np.maximum(cnts, 1)              # radial average power
        CumPr = np.cumsum(sums) / (sums.sum() + self.EPS)
        return bins, Pr, CumPr

    def _angle_anisotropy(self, P: np.ndarray) -> Tuple[float, Tuple[np.ndarray, np.ndarray], float]:
        h, w = P.shape
        cy, cx = (h-1)/2.0, (w-1)/2.0
        yy, xx = np.indices(P.shape)
        ry = (yy - cy) / h
        rx = (xx - cx) / w
        r = np.sqrt(rx*rx + ry*ry)
        theta = np.arctan2(ry, rx)

        band = (r >= self.aniso_rmin) & (r <= self.aniso_rmax)
        t = theta[band].ravel()
        v = P[band].ravel()

        bins = np.linspace(-np.pi, np.pi, self.n_theta + 1)
        idx = np.digitize(t, bins) - 1
        sums = np.bincount(idx, weights=v, minlength=self.n_theta)
        peak = sums.max() + self.EPS
        med = np.median(sums) + self.EPS
        ratio = float(peak / med)
        p = sums / (sums.sum() + self.EPS)
        spec_entropy = float(-np.sum(p * np.log(p + self.EPS)))
        return ratio, (bins[:-1], sums), spec_entropy

    def _spectral_slope(self, bins: np.ndarray, Pr: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        m = (bins >= self.slope_fmin) & (bins <= self.slope_fmax) & (Pr > 0)
        x = np.log(bins[m] + self.EPS)
        y = np.log(Pr[m] + self.EPS)
        if len(x) < 5:
            return np.nan, (np.nan, np.nan)
        k, b = np.polyfit(x, y, 1)
        alpha = -k
        return float(alpha), (float(k), float(b))

    @staticmethod
    def _freq_indices_from_cum(bins: np.ndarray, CumPr: np.ndarray, levels=(0.5, 0.8, 0.95)) -> Dict[str, float]:
        out = {}
        for lv in levels:
            idx = np.searchsorted(CumPr, lv)
            idx = np.clip(idx, 0, len(bins)-1)
            out[f"f{int(lv*100)}"] = float(bins[idx])
        return out

    @staticmethod
    def _high_freq_ratio(bins: np.ndarray, Pr: np.ndarray, thr: float) -> float:
        total = Pr.sum() + 1e-12
        high = Pr[bins > thr].sum()
        return float(high / total)

    @staticmethod
    def _spectral_centroid(bins: np.ndarray, Pr: np.ndarray) -> float:
        num = (bins * Pr).sum()
        den = Pr.sum() + 1e-12
        return float(num / den)

    @staticmethod
    def _noise_floor_and_snr_like(
        bins: np.ndarray, Pr: np.ndarray, hi_band=(0.35, 0.5), lo_band=(0.05, 0.15)
    ) -> Tuple[float, float]:
        hi = (bins >= hi_band[0]) & (bins <= hi_band[1])
        lo = (bins >= lo_band[0]) & (bins <= lo_band[1])
        floor = float(np.median(Pr[hi])) if hi.any() else np.nan
        lo_mean = float(np.mean(Pr[lo])) if lo.any() else np.nan
        snr_like = float(lo_mean / (floor + 1e-12)) if (np.isfinite(floor) and np.isfinite(lo_mean)) else np.nan
        return floor, snr_like

    def compute_metrics(self, img: np.ndarray) -> Dict[str, float]:
        x = self._preprocess(img)
        P = self._power_spectrum(x)
        bins, Pr, CumPr = self._radial_stats(P)

        # Base metrics
        metrics = {}
        for thr in self.hfr_thrs:
            metrics[f"HFR@{thr:.2f}"] = self._high_freq_ratio(bins, Pr, thr=thr)
        metrics["spectral_centroid"] = self._spectral_centroid(bins, Pr)
        metrics.update(self._freq_indices_from_cum(bins, CumPr, levels=(0.5, 0.8, 0.95)))

        alpha, _line = self._spectral_slope(bins, Pr)
        metrics["alpha_1_over_f"] = alpha

        aniso_ratio, (_ang_bins, _ang_sum), ang_entropy = self._angle_anisotropy(P)
        metrics["anisotropy_ratio"] = aniso_ratio
        metrics["angle_entropy"] = ang_entropy

        noise_floor, snr_like = self._noise_floor_and_snr_like(bins, Pr)
        metrics["noise_floor"] = noise_floor
        metrics["SNR_like"] = snr_like

        return metrics

    # ---------------------- Aggregation / scoring ---------------------- #
    @staticmethod
    def _nanrobust(x: np.ndarray, fn):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        return fn(x) if x.size else np.nan

    @staticmethod
    def _norm01(x: float, lo: float, hi: float) -> float:
        if np.isnan(x) or np.isnan(lo) or np.isnan(hi):
            return np.nan
        return float(np.clip((x - lo) / (hi - lo + 1e-12), 0, 1))

    def summarize_df(self, df: pd.DataFrame) -> Dict[str, float]:
        """Aggregate the per-image metric table for one set of projections into a summary + CompositeScore."""
        summary: Dict[str, float] = {}
        cols = df.columns.tolist()

        for k in cols:
            col = df[k].values
            summary[f"{k}_median"] = self._nanrobust(col, np.median)
            summary[f"{k}_p10"]    = self._nanrobust(col, lambda x: np.percentile(x, 10))
            summary[f"{k}_p90"]    = self._nanrobust(col, lambda x: np.percentile(x, 90))
            summary[f"{k}_IQR"]    = self._nanrobust(col, lambda x: np.percentile(x, 75) - np.percentile(x, 25))
            summary[f"{k}_CV"]     = self._nanrobust(col, lambda x: np.std(x) / (np.mean(x) + 1e-12))

        # Composite score: central quality + worst case - stability penalty
        # Good-metric set = HFR@* + spectral_centroid + f80 + SNR_like
        good_keys = [c for c in cols if c.startswith("HFR@")] + self.GOOD_KEYS_BASE
        bad_keys  = [k for k in self.BAD_KEYS if k in cols]

        parts = []
        for k in good_keys:
            med = summary.get(f"{k}_median", np.nan)
            p10 = summary.get(f"{k}_p10", np.nan)
            p90 = summary.get(f"{k}_p90", np.nan)
            parts.append(self._norm01(med, p10, p90))  # center
            parts.append(self._norm01(p10, p10, p90))  # worst case still not too bad

        for k in bad_keys:
            med = summary.get(f"{k}_median", np.nan)
            p10 = summary.get(f"{k}_p10", np.nan)
            p90 = summary.get(f"{k}_p90", np.nan)
            v1 = self._norm01(med, p10, p90)
            v2 = self._norm01(p90, p10, p90)
            parts.append(1 - v1 if np.isfinite(v1) else np.nan)
            parts.append(1 - v2 if np.isfinite(v2) else np.nan)

        # Stability penalty: normalized average of IQR/CV across all columns
        stab_terms = []
        for k in cols:
            iqr = summary.get(f"{k}_IQR", np.nan)
            cv  = summary.get(f"{k}_CV", np.nan)
            stab_terms += [self._norm01(iqr, 0.0, 0.3), self._norm01(cv, 0.0, 0.3)]
        stability_penalty = np.nanmean(stab_terms)

        CompositeScore = float(np.nanmean(parts) - 0.5 * stability_penalty)
        summary["CompositeScore"] = CompositeScore
        return summary

    # ---------------------- IO / main entry point ---------------------- #
    @staticmethod
    def _read_gray_png(path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img

    def analyze_folder(
        self,
        folder: str,
        pattern: str = "*.png",
        save_csv: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Read all PNGs under `folder` and return:
          per_img_df: per-projection metrics (index = file name)
          summary:    aggregated statistics + CompositeScore
        """
        p = Path(folder)
        files = sorted(p.glob(pattern))
        rows = []
        names = []
        for f in files:
            img = self._read_gray_png(f)
            if img is None:
                continue
            # central crop
            h, w = img.shape
            if h > 400 or w > 400:
                crop_h = min(400, h)
                crop_w = min(400, w)
                start_y = (h - crop_h) // 2
                start_x = (w - crop_w) // 2
                img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
            m = self.compute_metrics(img)
            rows.append(m)
            names.append(f.name)

        if not rows:
            raise RuntimeError(f"No readable PNGs found under: {folder}")

        per_img_df = pd.DataFrame(rows, index=names)
        summary = self.summarize_df(per_img_df)

        if save_csv:
            per_img_df.to_csv(save_csv)

        return per_img_df, summary