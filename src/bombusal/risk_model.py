"""Risk-score model (Pillar C, MLP).

An adaptive-weight MLP that maps the frequency-domain quality features of a sample's 2D renders to a
predicted human-algorithm discrepancy, `risk_pred = <w(x), x>`, used to select the reliable subset.

Two entry points, both used via ``pipeline/08_risk_mlp.py``:

* ``load_model`` + ``predict`` — inference with the trained weights and fitted scaler. Deterministic (eval
  mode, no RNG); this is the default path and gives the same scores on the same data.
* ``train`` — refit the scaler and MLP from scratch and overwrite the saved artifacts. Needed only when the
  dataset changes (new samples with manual glomeruli counts).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib

# Columns of volume_summaries.csv that are NOT model features.
NON_FEATURE_COLS = [
    "species", "sample", "region", "renders_dir",
    "CompositeScore", "index_id", "CQS_global", "risk_global", "difference",
]


def feature_columns(df):
    """Feature columns in a volume_summaries dataframe (everything except metadata/labels)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


class MLPWeighting(nn.Module):
    """Per-sample adaptive weighting: net(x) -> weights w; risk = <w, x>."""

    def __init__(self, in_dim=60, hidden=64, dropout=0.3, nonneg=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),  # output has the same dimension as the input (per-feature weights)
        )
        self.nonneg = nonneg
        self.softplus = nn.Softplus()

    def forward(self, x):
        w = self.net(x)
        if self.nonneg:
            w = self.softplus(w)
        risk = (w * x).sum(dim=1)
        return risk, w


# ----------------------------------------------------------------------------- inference

def load_model(models_dir):
    """Load the trained MLP, the fitted feature scaler, and the model config."""
    models_dir = Path(models_dir)
    bundle = joblib.load(models_dir / "mlp_config_scaler.joblib")
    cfg, scaler = bundle["config"], bundle["scaler"]
    model = MLPWeighting(in_dim=cfg["in_dim"], hidden=cfg["hidden"],
                         dropout=cfg["dropout"], nonneg=cfg["nonneg"])
    state = torch.load(models_dir / "mlp_weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, cfg


def predict(df, model, scaler, cfg):
    """Score a volume_summaries dataframe. Deterministic (eval mode, no RNG).

    Returns a copy of `df` with per-feature weight columns (``w_<feature>``) and ``risk_score_pred``.
    """
    feat = cfg["feature_cols"]
    X = scaler.transform(df[feat].to_numpy(dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        risk, w = model(torch.from_numpy(X))
    out = df.copy()
    weights = w.numpy()
    for i, col in enumerate(feat):
        out[f"w_{col}"] = weights[:, i]
    out["risk_score_pred"] = risk.numpy()
    return out


# ----------------------------------------------------------------------------- training

def _train_one_split(X_tr, y_tr, X_va, y_va, lr=1e-3, weight_decay=1e-3, epochs=200,
                     batch_size=16, lambda_wnorm=0.0, patience=30, nonneg=False, seed=42):
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(seed)
    model = MLPWeighting(in_dim=X_tr.shape[1], hidden=64, dropout=0.3, nonneg=nonneg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       batch_size=batch_size, shuffle=True)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va)

    best_state, best_val, patience_left = None, float("inf"), patience
    for _ in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            pred, w = model(xb)
            loss = loss_fn(pred, yb)
            if lambda_wnorm > 0:
                loss = loss + lambda_wnorm * (w.pow(2).mean())
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv, _ = model(Xv)
            val_mae = torch.mean(torch.abs(pv - yv)).item()
        if val_mae < best_val - 1e-6:
            best_val, best_state, patience_left = val_mae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv, _ = model(Xv)
        val_mae = torch.mean(torch.abs(pv - yv)).item()
    return model, pv.numpy(), val_mae


def train(df, models_dir, save=True):
    """Refit the scaler + MLP from scratch on `df` (rows with a `difference` label) and save the artifacts.

    Reports 5-fold CV MAE, then retrains a final model on all data (seed=999). Returns (model, scaler, cfg)
    ready for `predict`.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    df = df[df["difference"].notna()].copy()
    feat = feature_columns(df)
    X = StandardScaler().fit_transform(df[feat].to_numpy(dtype=np.float32)).astype(np.float32)
    y = df["difference"].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(df[feat].to_numpy(dtype=np.float32))  # keep a fitted scaler to save
    X = scaler.transform(df[feat].to_numpy(dtype=np.float32)).astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    val_maes = []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        _, _, mae = _train_one_split(X[tr], y[tr], X[va], y[va], lr=1e-3, weight_decay=1e-3,
                                     epochs=600, batch_size=8, lambda_wnorm=1e-3, patience=60, seed=fold * 111)
        val_maes.append(mae)
        print(f"[Fold {fold}] Val MAE: {mae:.4f}")
    print(f"[MLP] 5-fold mean Val MAE: {np.mean(val_maes):.4f} +/- {np.std(val_maes):.4f}")

    final_model, _, _ = _train_one_split(X, y, X, y, lr=1e-3, weight_decay=1e-3, epochs=800,
                                         batch_size=8, lambda_wnorm=1e-3, patience=80, seed=999)
    cfg = {"in_dim": X.shape[1], "hidden": 64, "dropout": 0.3, "nonneg": False, "feature_cols": feat}
    if save:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), models_dir / "mlp_weights.pt")
        joblib.dump({"config": cfg, "scaler": scaler}, models_dir / "mlp_config_scaler.joblib")
        print(f"Saved model + scaler to {models_dir}")
    return final_model, scaler, cfg
