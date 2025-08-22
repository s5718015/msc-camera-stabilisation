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
│   └── AblationModels/
│       ├── transformer_1head.keras
│       ├── transformer_1head_x_scaler.pkl
│       ├── transformer_1head_y_scaler.pkl
│       ├── transformer_no_posenc.keras
│       ├── transformer_no_posenc_x_scaler.pkl
│       └── transformer_no_posenc_y_scaler.pkl
│
├── Notebooks/
│   ├── compare_transformer_vs_others.ipynb
│   ├── train_transformer_model.ipynb
│   └── PriorWorksNotebooks/
│       ├── train_kalman_lstm_model.ipynb
│       └── train_model.ipynb
│
├── Code/
│   ├── transformer_animated_visual_demo.py   # animated XY comparison
│   ├── raw_vs_transformer_video_v2.py        # split-screen: RAW | Transformer (+ optional jitter)
│   ├── synthetic_data.py                     # synthetic metrics + plots
│   ├── real_data.py                          # real metrics (rolling windows vs Kalman)
│   └── utils_eval.py                         # shared helpers (windowing, curvature, scaling)
│
├── Report/
│   ├── CameraStabilizationReport.pdf
│   └── VideoDemo.mp4
│
├── requirements.txt
└── README.md
