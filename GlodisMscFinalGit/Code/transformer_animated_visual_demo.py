"""
transformer_animated_visual_demo_fixed.py
Animated plot comparing Kalman input, ground truth (smooth), LSTM, and Transformer predictions,
rendered frame-by-frame so the trajectories grow over time.

Data shape contract
-------------------
- X: (1, T, 3) as (t, x, y).
- Y: (T, 3) or (1, T, 3) as (t, x, y).
- Model outputs: (1, T, 3) or (1, T, 2). If (x,y) only, we handle slicing.

Reproducibility
---------------
- Visual demo focuses on presentation; seed if you sample subsequences.
- If scalers are used, inverse-transform the Transformer outputs and then take [:, 1:] for XY.

Example
-------
python transformer_animated_visual_demo_fixed.py --save results/demo.gif

References
----------
AI-ASSISTED DISCLOSURE (NCCA citation)
Tool: ChatGPT (OpenAI), used for brainstorming and small code suggestions.
Final code, experiments, and results are the author's own. Where AI assisted snippets
were adapted, a CITATION block appears near the relevant function.
Date: 2025-08-18

[1] Vaswani et al., 2017; [2] Hochreiter & Schmidhuber, 1997; [3] Kalman, 1960;
[6] NumPy; [7] Matplotlib;
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

LSTM_MODEL_PATH = "Models/hybrid_kalman_lstm_model.h5"
TRANSFORMER_MODEL_PATH = "Models/transformer_model_scaled.keras"
X_PATH = "Data/X_kalman.npy"
Y_PATH = "Data/Y_smooth.npy"
X_SCALER_PATH = "Models/transformer_x_scaler.pkl"
Y_SCALER_PATH = "Models/transformer_y_scaler.pkl"
SAVE_PATH ="" #"results/demo.gif"(so we dont save everytime)

def main():
    print("Loading models...")
    lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
    transformer_model = load_model(TRANSFORMER_MODEL_PATH, compile=False)
    print("Models loaded.")

    print("Loading scalers...")
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    print("Scalers loaded.")

    print("Loading data...")
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    print("Data loaded.")

    x = X[0:1]
    y_true = Y[0]
    y_lstm = lstm_model.predict(x, verbose=0)[0]
    
    x_original = X[0:1]  
    x_scaled = x_scaler.transform(x_original[0]).reshape(1, 124, 3)

    y_transformer_scaled = transformer_model.predict(x_scaled, verbose=0)[0] 

    y_transformer_unscaled = y_scaler.inverse_transform(y_transformer_scaled)

    y_transformer = y_transformer_unscaled[:, 1:]


    print("Predictions done.")

    print("Setting up plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    kalman_input = x[0][:, 1:]
    y_true_xy = y_true[:, 1:]

    all_data = np.concatenate([kalman_input, y_true_xy, y_lstm[:, 1:], y_transformer], axis=0)
    ax.set_xlim(np.min(all_data[:, 0]) - 0.1, np.max(all_data[:, 0]) + 0.1)
    ax.set_ylim(np.min(all_data[:, 1]) - 0.1, np.max(all_data[:, 1]) + 0.1)

    kalman_line, = ax.plot([], [], 'r--', label='Kalman Input')
    ground_line, = ax.plot([], [], 'g-', label='Ground Truth Smooth')
    lstm_line, = ax.plot([], [], 'b-', label='LSTM Prediction')
    transformer_line, = ax.plot([], [], 'm-', label='Transformer Prediction')
    ax.legend()
    # BEGIN CITATION 
    # Tool: ChatGPT (OpenAI) — date: 2025-08-19
    # Prompt (abridged): “Matplotlib FuncAnimation: update multiple Line2D objects per frame
    # from time-series slices (set_data on each line, using [:frame+1] indexing).”
    # Contribution: initial structure of the per-frame update loop.
    # Modifications by Glódís Ylja Hilmarsdóttir Kjærnested: variable naming, axis selection, frame bounds,
    # legend ordering, and integration with existing (t, x, y) layout.
    # END CITATION
    def update(frame):
        kalman_line.set_data(kalman_input[:frame+1, 0], kalman_input[:frame+1, 1])
        ground_line.set_data(y_true_xy[:frame+1, 0], y_true_xy[:frame+1, 1])
        lstm_line.set_data(y_lstm[:frame+1, 1], y_lstm[:frame+1, 2])
        transformer_line.set_data(y_transformer[:frame+1, 0], y_transformer[:frame+1, 1])
        return kalman_line, ground_line, lstm_line, transformer_line



    print("Creating animation...")
    ani = animation.FuncAnimation(fig, update, frames=124, interval=60, blit=False)
    print("Animation ready, displaying window...")
    plt.tight_layout()

# Save only if SAVE_PATH is set. otherwise just show
    if SAVE_PATH:
        try:
            Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving animation to {SAVE_PATH} ...")
            ani.save(SAVE_PATH)  
            print("Saved:", SAVE_PATH)
        except Exception as e:
            print("Could not save animation:", e)

    
    plt.show()

if __name__ == "__main__":
    main()



