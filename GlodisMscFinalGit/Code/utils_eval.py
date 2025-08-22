"""
utils_eval.py
Shared helpers for evaluation, metrics, simple smoothing refs, and timing.

"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator


def ensure_xy(arr: np.ndarray) -> np.ndarray:
    """
    Return a (T,2) XY array from common shapes:
    - (T,2) -> itself
    - (T,3+) -> columns 1:3 (assumes [t,x,y] or [...,x,y])
    - (1,T,D) or (N,T,D) -> take first batch then apply rule above

    Raises:
        ValueError if input doesn't look like time-series with 2+ features.
    """
    a = np.asarray(arr)
    if a.ndim == 3:  # (N,T,D)
        a = a[0]     # take first sequence

    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"ensure_xy expected (T,>=2) or (N,T,>=2); got {a.shape}")

    if a.shape[1] >= 3:
        return a[:, 1:3]  # take x,y assuming [t,x,y]
    return a[:, :2]       # already (x,y)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two (T,2) series."""
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"MSE requires same shape, got {a.shape} vs {b.shape}")
    return float(np.mean(np.sum((a - b) ** 2, axis=1)))


def curvature_series(xy: np.ndarray) -> np.ndarray:
    """
    Discrete curvature proxy for a 2D path (T,2).
    Useful as a smoothness score (lower is smoother).
    """
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("curvature_series expects (T,2)")
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2 + 1e-9) ** 1.5
    return np.abs(ddx * dy - dx * ddy) / denom


def inverse_transform_xy_with_any_scaler(scaler: BaseEstimator, xy_2: np.ndarray) -> np.ndarray:
    """
    Inverse-transform a (T,2) array even if the scaler was fit with >2 features.
    Pads with zeros to match scaler.n_features_in_, inverse-transforms, then returns x,y.
    """
    xy_2 = np.asarray(xy_2)
    n_feat = getattr(scaler, "n_features_in_", None)
    if n_feat is None:
        for attr in ("min_", "scale_", "data_min_", "data_max_"):
            if hasattr(scaler, attr):
                n_feat = len(getattr(scaler, attr))
                break
    if not n_feat or n_feat == 2:
        return scaler.inverse_transform(xy_2)

    pad = np.zeros((xy_2.shape[0], n_feat - 2), dtype=xy_2.dtype)
    merged = np.hstack([xy_2, pad])
    inv = scaler.inverse_transform(merged)
    return inv[:, :2]


def rolling_windows(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Turn (T,D) into overlapping windows (N, win, D) with stride 1.
    If T < win, the front is padded by repeating the first row.
    """
    a = np.asarray(arr)
    T = a.shape[0]
    if T < win:
        pad = np.repeat(a[:1], win - T, axis=0)
        a = np.vstack([pad, a])
        T = a.shape[0]
    N = T - win + 1
    return np.stack([a[i:i + win] for i in range(N)], axis=0)


def time_inference(model, x_batch: np.ndarray, repeats: int = 30) -> float:
    """
    Measure average latency of model.predict on x_batch (after a warmup).
    Returns seconds per call.
    """
    _ = model.predict(x_batch, verbose=0)  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(x_batch, verbose=0)
    return float((time.perf_counter() - t0) / repeats)


def load_transformer_scalers(x_scaler_path: str, y_scaler_path: str):
    """
    Load the X and Y scalers via joblib with clear error messages.
    Returns (x_scaler, y_scaler).
    """
    import joblib, os
    if not os.path.exists(x_scaler_path):
        raise FileNotFoundError(f"X scaler not found: {x_scaler_path}")
    if not os.path.exists(y_scaler_path):
        raise FileNotFoundError(f"Y scaler not found: {y_scaler_path}")
    try:
        x_scaler = joblib.load(x_scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load X scaler at {x_scaler_path}: {e}")
    try:
        y_scaler = joblib.load(y_scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Y scaler at {y_scaler_path}: {e}")
    return x_scaler, y_scaler
