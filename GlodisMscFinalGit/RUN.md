# RUN.md — How to Run the MSc Project

This guide reproduces the figures/metrics shown in the video and thesis.  
Repo layout assumed:
```
Code/  Data/  Models/  Report/  README.md  requirements.txt
```

## 1) Environment

```bash
# Create & activate a clean env (example with venv)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Note: If running headless (no GUI), Matplotlib will still save PNGs. If you see a display error, run with `--no-plots`.

## 2) Check files exist

```bash
# Models
ls Models/transformer_model_scaled.keras Models/hybrid_kalman_lstm_model.h5    Models/transformer_x_scaler.pkl Models/transformer_y_scaler.pkl

# Data
ls Data/X_kalman.npy Data/Y_smooth.npy Data/handheld_camera_data_numpy.npy
```

If any path differs, update the constants at the top of the scripts or rename the files.

## 3) Synthetic evaluation (metrics + PNGs)

```bash
python Code/synthetic_data.py --thesis-metrics
# Optional flags:
#   --no-plots                      # skip PNGs (fast)
#   --curv-smooth 61 --curv-arclen 64
#   --scale-invariant               # shape-only MSE
```

Outputs:
- `results/metrics_synthetic_thesis.json` and `.csv`
- `results/synthetic_{mse|curv|lat}_thesis.png` (if not using `--no-plots`)

## 4) Real-data evaluation (rolling windows vs Kalman reference)

```bash
python Code/real_data.py --thesis-metrics   --auto-align --calibrate --ref-range-warp   --curv-excess --curv-excess-clamp   --curv-smooth 61 --curv-smooth-models-only --curv-arclen 64
```

Outputs:
- `results/metrics_real_rolling_3way_thesis.json` and `.csv`  
Note: On real data, Kalman is the deterministic reference → **Kalman MSE = 0**; emphasise curvature + latency.

## 5) Animated XY demo (optional)

```bash
# Saves a short GIF (use .mp4 if ffmpeg is installed)
python Code/transformer_animated_visual_demo.py --save results/demo.gif
```

## 6) RAW vs Transformer split-screen video (optional)

```bash
python Code/raw_vs_transformer_video_v2.py   --video Data/Input/Input.mp4   --output results/transformer_vs_raw_jitter.mp4   --x-path Data/X_kalman.npy   --y-path Data/Y_smooth.npy   --transformer-model Models/transformer_model_scaled.keras   --x-scaler Models/transformer_x_scaler.pkl   --y-scaler Models/transformer_y_scaler.pkl   --gain 1.0 --jitter-std 2.0
```

---

## Options quick notes

- `--curv-smooth K` : moving-average smoothing window (odd K) applied **only** for curvature calculation.  
- `--curv-arclen M` : resample each window to **M** points by arc length (curvature only).  
- `--auto-align` / `--seq-shift {0,1}` : align predictions to the Kalman reference for fair comparison.  
- `--calibrate` : affine fit predictions to reference before metrics (like-for-like scale/offset).

---

## Troubleshooting

- **Matplotlib display error (headless)** → add `--no-plots` to synthetic run, or ensure your environment supports a GUI.  
- **ffmpeg not found** → save the animation as `.gif` instead of `.mp4`, or install ffmpeg.  
- **TensorFlow `safe_mode` error** → if present in your local scripts, remove `safe_mode=False` in `load_model(...)`.  
- **Path/case mismatch** → this repo uses `Models/`, `Data/`, `Code/` with capital letters; keep paths exact on Linux/macOS.

---

## Reproducibility note

- Runs are single-threaded for latency stability.  
- Scalers are saved and used for (inverse) transforms.  
- The thesis tables/figures are reproduced by Sections 3–4 above.
