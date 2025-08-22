"""
AI-ASSISTED DISCLOSURE (NCCA citation)
Tool: ChatGPT (OpenAI), used for brainstorming and small code suggestions.
Final code, experiments, and results are the author's own. Where AI-assisted snippets
were adapted, a CITATION block appears near the relevant function.
Date: 2025-08-19
"""


import os, json, time, argparse, warnings
from pathlib import Path
import joblib, numpy as np, matplotlib.pyplot as plt
from sklearn.exceptions import InconsistentVersionWarning
from tensorflow.keras.models import load_model

# Config 
REAL_SEQ_PATH          = "Data/handheld_camera_data_numpy.npy"   
LSTM_MODEL_PATH        = "Models/hybrid_kalman_lstm_model.h5"
TRANSFORMER_MODEL_PATH = "Models/transformer_model_scaled.keras"
X_SCALER_PATH          = "Models/transformer_x_scaler.pkl"
Y_SCALER_PATH          = "Models/transformer_y_scaler.pkl"

OUTDIR = Path("results"); OUTDIR.mkdir(exist_ok=True)
WIN = 124

# Single thread CPU for latency stability
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","0")
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Helpers
def ensure_xy(arr):
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[-1] > 2: return arr[..., :2]
    return arr
# BEGIN CITATION
# Tool: ChatGPT (OpenAI) — date: 2025-08-19
# Prompt (abridged): “Resample a 2D polyline to m points equally spaced by arc length using NumPy.”
# Contribution: initial structure for cumulative arc-length + np.interp resampling.
# Modifications by Glódís Ylja Hilmarsdóttir Kjærnested: windowed (N,WIN,2) batching, zero-length guard,
# dtype handling, and integration with per-window curvature evaluation.
# END CITATION
def resample_arclength_windows(seq_win, m):
    """Resample each [WIN,2] polyline to 'm' points equally spaced by arc length. seq_win: [N,WIN,2] -> [N,m,2]"""
    seq_win = np.asarray(seq_win, dtype=float)
    N, W, _ = seq_win.shape
    out = np.empty((N, m, 2), dtype=float)
    tgt = np.linspace(0.0, 1.0, m)
    for i in range(N):
        xy = seq_win[i]
        dx = np.diff(xy[:,0]); dy = np.diff(xy[:,1])
        seg = np.sqrt(dx*dx + dy*dy)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-12:
            out[i] = np.repeat(xy[:1], m, axis=0); continue
        s /= s[-1]
        out[i,:,0] = np.interp(tgt, s, xy[:,0])
        out[i,:,1] = np.interp(tgt, s, xy[:,1])
    return out

def curvature_series(xy):
    xy = np.asarray(xy, dtype=float)
    dx = np.gradient(xy[:,0]); dy = np.gradient(xy[:,1])
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    return np.abs(ddx*dy - dx*ddy) / (dx**2 + dy**2 + 1e-9)**1.5

def mse_windows(pred_win, ref_win):
    se = np.sum((pred_win - ref_win)**2, axis=2)     
    return float(np.mean(np.mean(se, axis=1)))       # mean over frames then windows
# BEGIN CITATION 
# Tool: ChatGPT (OpenAI) — date: 2025-08-19
# Prompt (abridged): “Given a scikit-learn scaler fitted with ≥3 features, how to inverse_transform only (x,y)?”
# Contribution: idea to pad dummy columns to match scaler.n_features_in_ then slice back to XY.
# Modifications by Glódís Ylja Hilmarsdóttir Kjærnested: robust feature-count detection across scaler types,
# vectorised reshape for sequences, dtype preservation.
# END CITATION
def inverse_transform_xy_with_any_scaler(scaler, xy2):
    xy2 = np.asarray(xy2)
    n_feat = getattr(scaler, "n_features_in_", None)
    if n_feat is None:
        for attr in ("min_","scale_","data_min_","data_max_"):
            if hasattr(scaler, attr): n_feat = len(getattr(scaler, attr)); break
    if not n_feat or n_feat == 2: return scaler.inverse_transform(xy2)
    pad = np.zeros((xy2.shape[0], n_feat-2), dtype=xy2.dtype)
    merged = np.hstack([xy2, pad]); inv = scaler.inverse_transform(merged)
    return inv[:, :2]

def inverse_transform_xy_seq(scaler, seq):  
    N,W,D = seq.shape; flat = seq.reshape(-1,D)
    inv = inverse_transform_xy_with_any_scaler(scaler, flat)
    return inv.reshape(N,W,D)

def rolling_windows(arr, win):
    arr = np.asarray(arr); T = arr.shape[0]; N = T - win + 1
    strides = (arr.strides[0],) + arr.strides
    shape = (N, win) + arr.shape[1:]
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
# BEGIN CITATION (AI-ASSISTED)
# Tool: ChatGPT (OpenAI) — date: 2025-08-19
# Prompt (abridged): “Minimal constant-velocity Kalman filter for 2D position: define F, H, Q(dt), R and iterate.”
# Contribution: high-level structure of prediction/update with (x,y,vx,vy) state.
# Modifications by Glódís Ylja Hilmarsdóttir Kjærnested: tuned Q/R scaling, per-step allocation, dtype control,
# and returning position-only series for use as a deterministic real-data reference.
# END CITATION
def kalman_reference(meas_xy, dt=1.0, q=1e-3, r=1e-2):
    meas_xy = np.asarray(meas_xy, dtype=float); T = meas_xy.shape[0]
    xhat = np.zeros((T,4)); P = np.eye(4)
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    H = np.array([[1,0,0,0],[0,1,0,0]], float)
    Q = q * np.block([[dt**4/4*np.eye(2), dt**3/2*np.eye(2)],
                      [dt**3/2*np.eye(2), dt**2*np.eye(2)]])
    R = r * np.eye(2)
    for t in range(T):
        xhat[t] = (F @ (xhat[t-1] if t>0 else np.zeros(4))).ravel()
        P = F @ P @ F.T + Q
        y = meas_xy[t] - (H @ xhat[t]); S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        xhat[t] = xhat[t] + (K @ y); P = (np.eye(4) - K @ H) @ P
    return xhat[:, :2]

def normalise_windows_to_ref(pred_win, ref_win):
    mins = ref_win.min(axis=1); maxs = ref_win.max(axis=1)
    span = np.clip(maxs - mins, 1e-9, None)
    pred_n = (pred_win - mins[:,None,:]) / span[:,None,:]
    ref_n  = (ref_win  - mins[:,None,:]) / span[:,None,:]
    return pred_n, ref_n

def minmax_windows(seq_win):
    mn = seq_win.min(axis=1, keepdims=True); mx = seq_win.max(axis=1, keepdims=True)
    span = np.clip(mx - mn, 1e-9, None)
    return (seq_win - mn) / span

def warp_to_ref_range(pred_win, ref_win):
    """Map pred window min/max to reference window min/max per axis."""
    pmin = pred_win.min(axis=1); pmax = pred_win.max(axis=1)
    rmin = ref_win.min(axis=1);  rmax = ref_win.max(axis=1)
    pspan = np.clip(pmax - pmin, 1e-9, None)
    pred01 = (pred_win - pmin[:, None, :]) / pspan[:, None, :]
    return rmin[:, None, :] + pred01 * (rmax - rmin)[:, None, :]

def align_pair(pred_win, ref_win, shift):
    if shift == 0: return pred_win, ref_win
    if shift > 0:  return pred_win[:, shift:, :], ref_win[:, :-shift, :]
    return pred_win, ref_win

def pick_best_shift(pred_win, ref_win):
    p0,r0 = align_pair(pred_win, ref_win, 0); p1,r1 = align_pair(pred_win, ref_win, 1)
    last0 = p0[:,-1,:] - r0[:,-1,:]; last1 = p1[:,-1,:] - r1[:,-1,:]
    mse0 = float(np.mean(np.sum(last0**2, axis=1))); mse1 = float(np.mean(np.sum(last1**2, axis=1)))
    return 1 if mse1 < mse0 else 0

def time_inference_per_window(model, Xw, sample_idx=None):
    if sample_idx is None: sample_idx = np.arange(Xw.shape[0])
    sample_idx = np.asarray(sample_idx)
    for i in sample_idx[:min(5,len(sample_idx))]: _ = model.predict(Xw[i:i+1], verbose=0)
    t0 = time.perf_counter()
    for i in sample_idx: _ = model.predict(Xw[i:i+1], verbose=0)
    t1 = time.perf_counter()
    return float((t1 - t0) / len(sample_idx))

# Affine calibration 
def affine_fit_2d(pred_win, ref_win, max_samples=20000, seed=0):
    rng = np.random.default_rng(seed)
    N, Wlen, _ = pred_win.shape
    total = N*Wlen
    k = min(max_samples, total)
    sel = rng.choice(total, size=k, replace=False)
    pred = pred_win.reshape(-1, 2)[sel]
    ref  = ref_win.reshape(-1, 2)[sel]
    X = np.c_[pred, np.ones((k,1))]
    wx, *_ = np.linalg.lstsq(X, ref[:,0], rcond=None)
    wy, *_ = np.linalg.lstsq(X, ref[:,1], rcond=None)
    W = np.vstack([wx, wy]).T
    return W  

def apply_affine(W, seq_win):
    N, Wlen, _ = seq_win.shape
    X = np.c_[seq_win.reshape(-1,2), np.ones((N*Wlen,1))]
    Y = X @ W
    return Y.reshape(N, Wlen, 2)

# Smoothing only for curvature
def smooth_xy_windows(seq_win, k):
    if k is None or k <= 1: return seq_win
    if k % 2 == 0: k += 1
    kernel = np.ones(k, dtype=float) / k
    N, W, D = seq_win.shape
    out = np.empty_like(seq_win)
    pad = k // 2
    for d in range(D):
        a = np.pad(seq_win[:, :, d], ((0, 0), (pad, pad)), mode="edge")
        out[:, :, d] = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, a)
    return out

def window_mean_curv_list(seq_win):
    """Return mean curvature per window as a vector [N]."""
    return np.array([np.mean(curvature_series(seq_win[i])) for i in range(seq_win.shape[0])], dtype=float)

# Main 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thesis-metrics", action="store_true",
                        help="Compute MSE/curvature on per-window [0,1] XY (and single-thread latency).")
    parser.add_argument("--ref-range-warp", action="store_true",
                        help="Map each prediction window to the reference window's XY min/max before metrics.")
    parser.add_argument("--seq-shift", type=int, default=1, choices=[0,1],
                        help="Align predictions to reference by this many frames (default: 1).")
    parser.add_argument("--auto-align", action="store_true",
                        help="Auto-pick shift {0,1} per-model that best matches the reference.")
    parser.add_argument("--curv-excess", action="store_true",
                        help="Report curvature as (model − Kalman) per-window mean (Kalman=0).")
    parser.add_argument("--curv-excess-clamp", action="store_true",
                        help="Clamp per-window excess curvature at ≥ 0 before averaging (never negative).")

    parser.add_argument("--fast-latency", action="store_true",
                        help="Estimate latency using a subset of windows (much faster).")
    parser.add_argument("--latency-samples", type=int, default=300,
                        help="Number of windows to sample when using --fast-latency (default: 300).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plotting to save time.")
    parser.add_argument("--scale-invariant", action="store_true",
                        help="Normalise each window independently (shape-only).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit an affine map from model outputs to the Kalman reference before metrics.")
    parser.add_argument("--curv-smooth", type=int, default=1,
                        help="Odd window for moving-average smoothing applied ONLY before curvature (1 = off).")
    parser.add_argument("--curv-arclen", type=int, default=0,
                        help="If >0, resample each window to this many points by arc length for curvature only.")
    parser.add_argument("--curv-smooth-models-only", action="store_true",
                        help="If set, smooth only LSTM/Transformer for curvature; leave Kalman unsmoothed.")
    args = parser.parse_args()
    suffix = "_thesis" if args.thesis_metrics else ""

    # Load sequence + references
    seq = np.load(REAL_SEQ_PATH)                      
    meas_xy = ensure_xy(seq)                          
    y_ref_full = kalman_reference(meas_xy)            

    # Windows
    Xw = rolling_windows(seq, WIN)                    
    y_ref_win = rolling_windows(y_ref_full, WIN)      
    N = Xw.shape[0]

    # Models & scalers
    lstm = load_model(LSTM_MODEL_PATH, compile=False, safe_mode=False)
    transformer = load_model(TRANSFORMER_MODEL_PATH, compile=False, safe_mode=False)
    x_scaler = joblib.load(X_SCALER_PATH); y_scaler = joblib.load(Y_SCALER_PATH)

    # Predictions
    y_lstm_seq = ensure_xy(lstm.predict(Xw, verbose=0))     
    Xw_scaled = x_scaler.transform(Xw.reshape(-1, Xw.shape[-1])).reshape(Xw.shape)
    y_tr_seq_s = ensure_xy(transformer.predict(Xw_scaled, verbose=0))   # scaler space
    y_tr_seq   = inverse_transform_xy_seq(y_scaler, y_tr_seq_s)         # original XY

    # Alignment 
    shift_l = pick_best_shift(y_lstm_seq, y_ref_win) if args.auto_align else args.seq_shift
    shift_t = pick_best_shift(y_tr_seq,  y_ref_win)  if args.auto_align else args.seq_shift
    y_ref_al_l = align_pair(y_ref_win, y_ref_win, shift_l)[1]
    y_lstm_al  = align_pair(y_lstm_seq, y_ref_win, shift_l)[0]
    y_tr_al    = align_pair(y_tr_seq,   y_ref_win, shift_t)[0]
    # Use the reference aligned to LSTM shift for metrics
    y_ref_al = y_ref_al_l
    minW = min(y_ref_al.shape[1], y_lstm_al.shape[1], y_tr_al.shape[1])
    y_ref_al  = y_ref_al[:, :minW, :]
    y_lstm_al = y_lstm_al[:, :minW, :]
    y_tr_al   = y_tr_al[:, :minW, :]

    # affine calibration to reference (affects MSE & curvature)
    if args.calibrate:
        W_l = affine_fit_2d(y_lstm_al, y_ref_al)
        W_t = affine_fit_2d(y_tr_al,   y_ref_al)
        y_lstm_al = apply_affine(W_l, y_lstm_al)
        y_tr_al   = apply_affine(W_t, y_tr_al)

    
    if args.ref_range_warp:
        y_lstm_al = warp_to_ref_range(y_lstm_al, y_ref_al)
        y_tr_al   = warp_to_ref_range(y_tr_al,   y_ref_al)

    
    if args.thesis_metrics:
        # Normalisation choice
        if args.scale_invariant:
            ref_n   = minmax_windows(y_ref_al)
            lstm_n  = minmax_windows(y_lstm_al)
            tr_n    = minmax_windows(y_tr_al)
        else:
            lstm_n, ref_n = normalise_windows_to_ref(y_lstm_al, y_ref_al)
            tr_n,   _     = normalise_windows_to_ref(y_tr_al,   y_ref_al)

        # Curvature prep
        if args.curv_smooth_models_only:
            ref_for_curv  = ref_n
            lstm_for_curv = smooth_xy_windows(lstm_n, args.curv_smooth)
            tr_for_curv   = smooth_xy_windows(tr_n,   args.curv_smooth)
            if args.curv_arclen and args.curv_arclen > 1:
                lstm_for_curv = resample_arclength_windows(lstm_for_curv, args.curv_arclen)
                tr_for_curv   = resample_arclength_windows(tr_for_curv,   args.curv_arclen)
        else:
            ref_for_curv  = smooth_xy_windows(ref_n,  args.curv_smooth)
            lstm_for_curv = smooth_xy_windows(lstm_n, args.curv_smooth)
            tr_for_curv   = smooth_xy_windows(tr_n,   args.curv_smooth)
            if args.curv_arclen and args.curv_arclen > 1:
                ref_for_curv  = resample_arclength_windows(ref_for_curv,  args.curv_arclen)
                lstm_for_curv = resample_arclength_windows(lstm_for_curv, args.curv_arclen)
                tr_for_curv   = resample_arclength_windows(tr_for_curv,   args.curv_arclen)

        # Sanity print
        print("THESIS MODE ranges:",
              "ref",  (float(ref_n.min()),  float(ref_n.max())),
              "lstm", (float(lstm_n.min()), float(lstm_n.max())),
              "tr",   (float(tr_n.min()),   float(tr_n.max())),
              "| shifts (L,T) =", (shift_l, shift_t),
              "| calibrate =", args.calibrate,
              "| scale_invariant =", args.scale_invariant,
              "| ref_range_warp =", args.ref_range_warp,
              "| curv_smooth =", args.curv_smooth,
              "| curv_models_only =", args.curv_smooth_models_only,
              "| curv_arclen =", args.curv_arclen,
              "| curv_excess =", args.curv_excess,
              "| curv_excess_clamp =", args.curv_excess_clamp)

        # MSE over FULL windows
        mse_kf    = 0.0
        mse_lstm  = mse_windows(lstm_n, ref_n)
        mse_trans = mse_windows(tr_n,   ref_n)

        # Curvature: absolute or 'excess vs Kalman' with optional clamp
        curv_ref_list  = window_mean_curv_list(ref_for_curv)
        curv_lstm_list = window_mean_curv_list(lstm_for_curv)
        curv_tr_list   = window_mean_curv_list(tr_for_curv)

        if args.curv_excess:
            diff_l = curv_lstm_list - curv_ref_list
            diff_t = curv_tr_list   - curv_ref_list
            if args.curv_excess_clamp:
                diff_l = np.maximum(diff_l, 0.0)
                diff_t = np.maximum(diff_t, 0.0)
            curv_kf    = 0.0
            curv_lstm  = float(np.mean(diff_l))
            curv_trans = float(np.mean(diff_t))
        else:
            curv_kf    = float(np.mean(curv_ref_list))
            curv_lstm  = float(np.mean(curv_lstm_list))
            curv_trans = float(np.mean(curv_tr_list))

        # Latency
        if args.fast_latency:
            S = min(N, max(1, int(args.latency_samples)))
            sample_idx = np.linspace(0, N-1, num=S, dtype=int)
        else:
            sample_idx = np.arange(N)
        lat_lstm  = time_inference_per_window(lstm,        Xw,        sample_idx)
        lat_trans = time_inference_per_window(transformer, Xw_scaled, sample_idx)
        lat_kf    = 0.0
    else:
        # Legacy last frame raw metrics
        y_lstm_last = y_lstm_seq[:, -1, :]; y_tr_last = y_tr_seq[:, -1, :]; y_ref_last = y_ref_win[:, -1, :]
        mse_last = lambda a,b: float(np.mean(np.sum((a-b)**2, axis=1)))
        mse_kf = 0.0; mse_lstm = mse_last(y_lstm_last, y_ref_last); mse_trans = mse_last(y_tr_last, y_ref_last)
        curv_kf = float(np.mean(curvature_series(y_ref_last)))
        curv_lstm = float(np.mean(curvature_series(y_lstm_last)))
        curv_trans = float(np.mean(curvature_series(y_tr_last)))
        def time_batched(model, X): _=model.predict(X[:2],0); t0=time.perf_counter(); _=model.predict(X,0); t1=time.perf_counter(); return float((t1-t0)/X.shape[0])
        lat_lstm = time_batched(lstm, Xw); lat_trans = time_batched(transformer, Xw_scaled); lat_kf = 0.0

    res3 = {"Kalman":{"mse":mse_kf,"curv":curv_kf,"lat":lat_kf},
            "LSTM":{"mse":mse_lstm,"curv":curv_lstm,"lat":lat_lstm},
            "Transformer":{"mse":mse_trans,"curv":curv_trans,"lat":lat_trans}}

    # Save
    with open(OUTDIR/f"metrics_real_rolling_3way{suffix}.json","w") as f: json.dump(res3,f,indent=2)
    with open(OUTDIR/f"metrics_real_rolling_3way{suffix}.csv","w") as f:
        f.write("Model,MSE,Curvature,Latency\n")
        for k in ["Kalman","LSTM","Transformer"]:
            r=res3[k]; f.write(f"{k},{r['mse']},{r['curv']},{r['lat']}\n")

    # Console
    print("\n=== Real Data (Rolling Windows) — 3-way Metrics ===")
    print(f"{'Model':<12} {'MSE':>12} {'Curvature':>12} {'Latency(s)':>12}")
    for k in ["Kalman","LSTM","Transformer"]:
        r=res3[k]; print(f"{k:<12} {r['mse']:>12.6f} {r['curv']:>12.6f} {r['lat']:>12.6f}")
    print("\nSaved to:", OUTDIR.resolve())

if __name__ == "__main__":
    main()

