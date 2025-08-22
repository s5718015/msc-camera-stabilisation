"""
raw_vs_transformer_video_v2.py
adds optional synthetic jitter to the RAW (left) side
for clearer visual contrast. Right side applies Transformer-based stabilization.

Data shape contract
-------------------
- X: (T, 3) as (t, x, y). Or (N, T, 3), then select one sequence (T, 3).
- Transformer output: (T, 3) or (T, 2). If (T, 2), prepend a zero time column (t,x,y)
  before inverse-scaling with y_scaler.

Reproducibility
---------------
- Use joblib.load for scalers; ignore version warnings if features align.
- Synthetic jitter uses a fixed RNG seed to make demos re-runnable.


References
----------
[1] Vaswani et al., 2017; [5] Keras/TensorFlow docs; [6] NumPy; [7] scikit-learn;
OpenCV documentation (affine warps).
"""

import argparse
import numpy as np
import cv2
import joblib
import tensorflow as tf


def inverse_transform_xy_with_any_scaler(scaler, arr_3: np.ndarray) -> np.ndarray:
    """
    Inverse-transform predictions that were scaled with a 3-feature scaler (t,x,y).
    Returns only (x,y) after inverse-scaling.
    """
    arr_3 = np.asarray(arr_3)
    inv = scaler.inverse_transform(arr_3)   
    return inv[:, 1:3]                      


def compute_smooth_xy(x_seq, scalers, transformer_model):
    """
    Return (raw_xy, smooth_xy) where:
      - raw_xy comes directly from X (t,x,y) -> (x,y)
      - smooth_xy is transformer prediction inverse-scaled via y_scaler, then (x,y)
    """
    x_seq = np.asarray(x_seq)               
    raw_xy = x_seq[:, 1:3]                  

    
    y_pred_scaled = transformer_model.predict(x_seq[None, ...], verbose=0).squeeze()
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    if y_pred_scaled.shape[1] == 2:
        T = y_pred_scaled.shape[0]
        y_pred_scaled = np.column_stack([np.zeros(T), y_pred_scaled])  # (T,3)

    smooth_xy = inverse_transform_xy_with_any_scaler(scalers["y"], y_pred_scaled)  # (T,2)
    return raw_xy, smooth_xy


def make_writer(path: str, w: int, h: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if vw.isOpened():
        return vw
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(path.rsplit(".", 1)[0] + ".avi", fourcc, fps, (w, h))


def render_split(video_path, output_path, raw_xy, smooth_xy, gain=1.0, jitter_std=2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    T = min(N, raw_xy.shape[0], smooth_xy.shape[0])
    raw_xy, smooth_xy = raw_xy[:T], smooth_xy[:T]

    delta = (smooth_xy - raw_xy) * float(gain)

    out = make_writer(output_path, W * 2, H, fps)
    rng = np.random.default_rng(123)

    for i in range(T):
        ok, frame = cap.read()
        if not ok:
            break

        
        left = frame.copy()
        jx, jy = rng.normal(0.0, jitter_std, size=2)  
        Mj = np.float32([[1, 0, jx], [0, 1, jy]])
        left = cv2.warpAffine(left, Mj, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    
        dx, dy = float(delta[i, 0]), float(delta[i, 1])
        Ms = np.float32([[1, 0, dx], [0, 1, dy]])
        right = cv2.warpAffine(frame, Ms, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    
        cv2.putText(left, "Raw + jitter", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
        cv2.putText(right, "Transformer", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)

        out.write(np.hstack([left, right]))

    cap.release()
    out.release()
    print(f" Wrote: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", default="results/transformer_vs_raw_jitter.mp4")
    ap.add_argument("--x-path", required=True)
    ap.add_argument("--y-path", required=True)
    ap.add_argument("--transformer-model", required=True)
    ap.add_argument("--x-scaler", required=True)
    ap.add_argument("--y-scaler", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--jitter-std", type=float, default=2.0, help="Std dev of synthetic jitter on RAW")
    args = ap.parse_args()

    X = np.load(args.x_path)
    if X.ndim == 3:
        x_seq = X[min(args.sample_index, X.shape[0] - 1)]
    else:
        x_seq = X

    scalers = {
        "x": joblib.load(args.x_scaler),
        "y": joblib.load(args.y_scaler),
    }
    transformer_model = tf.keras.models.load_model(args.transformer_model, compile=False)

    raw_xy, smooth_xy = compute_smooth_xy(x_seq, scalers, transformer_model)
    render_split(args.video, args.output, raw_xy, smooth_xy, gain=args.gain, jitter_std=args.jitter_std)


if __name__ == "__main__":
    main()
