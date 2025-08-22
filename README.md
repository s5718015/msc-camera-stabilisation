# MSc Project: Real-Time Camera Motion Smoothing

This repository contains the code and experiments for the MSc project on **Real-Time Camera Motion Smoothing for Virtual Production: A Comparative Study of Kalman, LSTM and Transformer Models**.  
The project investigates and compares three approaches:

**Author:** Glodis Ylja Hilmarsdottir Kjaernested  
**Institution:** Bournemouth University — Artificial Intelligence for Media (MSc)
**Supervisor:** Dr. Nait-Charif Hammadi  

- **Kalman filter** (classical baseline)
- **Hybrid Kalman + LSTM model**
- **Transformer-based smoothing model**

Kalman and Hybrid LSTM were previosly written by the Author and are used for comparison in this research. 

Both synthetic and real handheld camera motion data were used for evaluation.  
Metrics include **Mean Squared Error (MSE)**, **curvature smoothness**, and **latency**.

---

##  Project Structure
Project_Submission/
├── Models/
│   ├── hybrid_kalman_lstm_model.h5
│   ├── transformer_model_scaled.keras
│   ├── transformer_x_scaler.pkl
│   └── transformer_y_scaler.pkl
│
├── Data/
│   ├── X_kalman.npy
│   ├── Y_smooth.npy
│   ├── X_noisy_200.npy
│   ├── Y_smooth_200.npy
│   ├── handheld_camera_data_numpy.npy
│   └── Input
│       
│
├── AblationStudy/
│   ├── AblationCode/
│   │   ├── train_transformer_1head_model.py
│   │   └── transformer_no_positional_encoding.py
│   ├── AblationModels/
│   │   ├── transformer_1head.keras
│   │   ├── transformer_1head_x_scaler.pkl
│   │   ├── transformer_1head_y_scaler.pkl
│   │   ├── transformer_no_posenc.keras
│   │   ├── transformer_no_posenc_x_scaler.pkl
│   │   └── transformer_no_posenc_y_scaler.pkl
│   └── Notebooks/
│       ├── compare_transformer_vs_others.ipynb
│       ├── train_transformer_model.ipynb
│       └── PriorWorksNotebooks/
│           ├── train_kalman_lstm_model.ipynb
│           └── train_model.ipynb
│
├── Code/
│   ├── transformer_animated_visual_demo.py
│   ├── raw_vs_transformer_video_v2.py
│   ├── synthetic_data.py
│   ├── real_data.py
│   └── utils_eval.py
│
├── Report/
│   ├── CameraStabilizationReport.pdf
│   └── VideoDemo.mp4
│
├── requirements.txt
└── README.md
                        


---

##  How to Run 

Run the Real Data 
python Code\real_data.py `
  --thesis-metrics `
  --auto-align --calibrate --ref-range-warp `
  --curv-excess --curv-excess-clamp `
  --curv-smooth 61 --curv-smooth-models-only --curv-arclen 64 `
 


Run the Synthetic Data
python Code\synthetic_data.py `
  --thesis-metrics `
  --seq-shift 1 `
  --curv-smooth 1 `



Run the Animated Demo 
python .\Code\transformer_animated_visual_demo.py



Run Side By Side Visuals
python .\Code\raw_vs_transformer_video_v2.py `
  --video ".\Data\Input.mp4" `
  --output ".\results\transformer_vs_raw_jitter.mp4" `
  --x-path ".\Data\X_kalman.npy" `
  --y-path ".\Data\Y_smooth.npy" `
  --transformer-model ".\Models\transformer_model_scaled.keras" `
  --x-scaler ".\Models\transformer_x_scaler.pkl" `
  --y-scaler ".\Models\transformer_y_scaler.pkl" `
  --sample-index 0 `
  --gain 1.0 `
  --jitter-std 2.0


---

## Dependencies

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- joblib
- moviepy (for video demos)

Install everything via:
```bash
pip install -r requirements.txt
```

---

## Results in Action

- **Side-by-side video demo** shows raw vs smoothed motion.
- **Animated plot demo** visualizes trajectories growing frame by frame.
- Evaluation scripts generate **metrics + plots** for synthetic and real data.

---

## References

### AI assistance disclosure
I used ChatGPT to brainstorm structure and occasional code snippets. 
All experiments, evaluation, and final results are my own. 
Where AI generated code was adapted, it is cited inline in comments with prompts and modifications.
Where third party libraries or prior methods are used, they are cited below.

Vaswani et al. (2017). Attention Is All You Need. NeurIPS.

Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.

Kingma & Ba (2015). Adam: A Method for Stochastic Optimization. ICLR.

Chollet et al. Keras / TensorFlow Documentation.

Van der Walt et al. (2011). The NumPy Array: A Structure for Efficient Numerical Computation.

Hunter (2007). Matplotlib: A 2D Graphics Environment.

scikit-learn User Guide: Feature scaling & persistence.

Keras API — tf.keras.models.load_model & save formats: official reference and saving guide. (Used for loading .h5 and .keras in evaluate_* and demos.)

Keras Utilities — Time-series windowing (e.g., TimeseriesGenerator / timeseries_dataset_from_array). (Background for rolling windows used in evaluate_real.py.)

scikit-learn API — StandardScaler / MinMaxScaler + inverse_transform. (Transformer inverse-scaling back to original units.)

Joblib Docs — joblib.dump / joblib.load, version compatibility notes. (Loading scaler .pkl files.)

NumPy Reference — numpy.gradient (discrete curvature building block). (Used in curvature_series smoothness proxy.)

Matplotlib Gallery/Docs — savefig DPI & bbox_inches='tight'. (Consistent, publication quality PNGs.)

Matplotlib Animation — FuncAnimation, FFMpegWriter, PillowWriter. (Animated demo export; MP4/GIF flow.)

OpenCV-Python Tutorials — Geometric transforms & cv2.warpAffine. (Frame translation for video side by side.)

OpenCV API — VideoWriter (FourCC, fps, size). (Creating split videos in raw_vs_transformer_video*.py.)

OpenCV Drawing — cv2.circle, cv2.putText. (Overlays/markers to visualise residual jitter.)

Python Stdlib — time.perf_counter (high-res timing). (Latency measurement helpers.)

Python argparse — Command-line flags & help. (CLI interfaces across scripts.)

FilterPy — Practical Kalman filter implementations in Python. (Coding reference for constant-velocity KF.)

pykalman — Alternative Python Kalman filter library & examples. (Background/validation for KF approach.)

TensorFlow Performance Tips — warm-ups, repeated inference for stable timing. (Justifies “warm up then average” timing method.)

Matplotlib Axes — ax.set_aspect('equal') and layout tips. (Ensures geometrically honest XY overlays.)
