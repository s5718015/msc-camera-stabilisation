"""
train_transformer_1head_model.py
Train a lightweight Transformer (single-head) on (t, x, y) motion sequences.
Handles scaling, model checkpointing, and metric logging.

Data shape contract
-------------------
- Training input X: (N, T, 3) as (t, x, y).
- Target Y:        (N, T, 3) as (t, x, y) unless ablation differs.
- Inference:       (1, T, 3).

Reproducibility
---------------
- Set seeds for numpy / tensorflow; record versions in logs.
- Save fitted scalers (x_scaler, y_scaler) with joblib for consistent inference.

Example
-------
python train_transformer_1head_model.py --models-dir Models --epochs 50


"""


import os, json, random, argparse, joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def mean_curvature_xy(traj_xy: np.ndarray) -> float:
    """
    Mean curvature proxy: mean L2 norm of discrete 2nd difference over time.
    traj_xy: (N, T, 2) or (T, 2) returns float
    """
    if traj_xy.ndim == 2:
        traj_xy = traj_xy[None, ...]
    v = np.diff(traj_xy, axis=-2)   
    a = np.diff(v, axis=-2)         
    curv = np.linalg.norm(a, axis=-1) 
    return float(curv.mean())

def mse_xy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))



def _posenc(length, depth):
    """Graph-safe sinusoidal PE. Returns (length, depth)."""
    length = tf.cast(length, tf.int32)
    depth  = tf.cast(depth,  tf.int32)
    half   = tf.maximum(depth // 2, 1)

    positions = tf.cast(tf.range(length)[:, tf.newaxis], tf.float32)     
    dims      = tf.cast(tf.range(half)[tf.newaxis, :], tf.float32)       

    angle_rates = tf.pow(10000.0, -dims / tf.cast(half, tf.float32))     
    angle_rads  = positions * angle_rates                                

    pe = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)     
    pad = tf.maximum(0, depth - tf.shape(pe)[1])
    pe  = tf.pad(pe, [[0, 0], [0, pad]])                                 
    pe  = pe[:, :depth]                                                
    return pe

class PositionalEncoding(layers.Layer):
    def call(self, x):
        # x: (B, T, D)
        T = tf.shape(x)[1]
        D = tf.shape(x)[2]
        pe = _posenc(T, D)
        pe = tf.cast(pe, x.dtype)
        return x + pe[tf.newaxis, :, :]  

    def compute_output_shape(self, input_shape):
        return input_shape


def transformer_block(x, d_model=128, ff_dim=256, num_heads=1, dropout=0.1, causal=True):
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x, use_causal_mask=causal)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-5)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    return x

def build_model(seq_len: int, in_dim: int = 3, d_model: int = 128, ff_dim: int = 256, num_heads: int = 1):
    inputs = keras.Input(shape=(seq_len, in_dim))      
    x = layers.Dense(d_model)(inputs)                  
    x = PositionalEncoding()(x)                       
    x = transformer_block(x, d_model=d_model, ff_dim=ff_dim, num_heads=num_heads, dropout=0.1, causal=True)
    x = transformer_block(x, d_model=d_model, ff_dim=ff_dim, num_heads=num_heads, dropout=0.1, causal=True)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(2)(x)                       
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mse"])
    return model



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-path", required=True, help="Path to noisy inputs .npy (N,T,3) -> [t,x,y]")
    ap.add_argument("--y-path", required=True, help="Path to smooth targets .npy (N,T,3) -> [t,x,y]")
    ap.add_argument("--seq-len", type=int, default=124)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    ensure_dir(args.models_dir)
    ensure_dir(args.results_dir)

    
    X = np.load(args.x_path)  
    Y = np.load(args.y_path)  
    assert X.ndim == 3 and Y.ndim == 3 and X.shape == Y.shape, "Expected (N,T,3) for X and Y"
    assert X.shape[1] == args.seq_len, f"Expected seq-len {args.seq_len}, got {X.shape[1]}"


    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.3, random_state=args.seed, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=args.seed, shuffle=True)

    
    in_dim = X.shape[2]  
    out_dim = 2          
    scaler_x = MinMaxScaler().fit(X_train.reshape(-1, in_dim))                
    scaler_y = MinMaxScaler().fit(Y_train[..., 1:3].reshape(-1, out_dim))     

    def tx_inputs(a):  return scaler_x.transform(a.reshape(-1, in_dim)).reshape(a.shape)
    def tx_targets(a): return scaler_y.transform(a[..., 1:3].reshape(-1, out_dim)).reshape(a.shape[0], a.shape[1], out_dim)

    X_train_s, X_val_s, X_test_s = tx_inputs(X_train), tx_inputs(X_val), tx_inputs(X_test)
    Y_train_s, Y_val_s, Y_test_s = tx_targets(Y_train), tx_targets(Y_val), tx_targets(Y_test)

    
    model = build_model(seq_len=args.seq_len, in_dim=in_dim, d_model=128, ff_dim=256, num_heads=1)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]
    model.fit(
        X_train_s, Y_train_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=args.epochs, batch_size=args.batch, verbose=2, callbacks=callbacks
    )

    
    Y_pred_s = model.predict(X_test_s, verbose=0) 
    Y_pred = scaler_y.inverse_transform(Y_pred_s.reshape(-1, out_dim)).reshape(Y_pred_s.shape)
    Y_true = scaler_y.inverse_transform(Y_test_s.reshape(-1, out_dim)).reshape(Y_test_s.shape)

    mse = mse_xy(Y_true, Y_pred)
    curv = mean_curvature_xy(Y_pred)

    print(f"[1-head PE] Test MSE: {mse:.6f} | Curvature: {curv:.6f}")

    
    model_path = os.path.join(args.models_dir, "transformer_1head.keras")
    xsc_path  = os.path.join(args.models_dir, "transformer_1head_x_scaler.pkl")
    ysc_path  = os.path.join(args.models_dir, "transformer_1head_y_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler_x, xsc_path)
    joblib.dump(scaler_y, ysc_path)

    metrics = {
        "ablation": "heads_1_vs_4",
        "model_path": model_path,
        "x_scaler": xsc_path,
        "y_scaler": ysc_path,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch": args.batch,
        "mse": round(mse, 8),
        "curvature": round(curv, 8),
    }
    with open(os.path.join(args.results_dir, "ablation_heads_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {model_path}\n[save] {xsc_path}\n[save] {ysc_path}\n[save] results/ablation_heads_metrics.json")

if __name__ == "__main__":
    main()


    
    X = np.load(args.x_path)  
    Y = np.load(args.y_path)  
    assert X.ndim == 3 and Y.ndim == 3 and X.shape == Y.shape, "Expected (N,T,3) for X and Y"
    assert X.shape[1] == args.seq_len, f"Expected seq-len {args.seq_len}, got {X.shape[1]}"

    
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.3, random_state=args.seed, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=args.seed, shuffle=True)

    
    in_dim = X.shape[2] 
    out_dim = 2          
    scaler_x = MinMaxScaler().fit(X_train.reshape(-1, in_dim))         
    scaler_y = MinMaxScaler().fit(Y_train[..., 1:3].reshape(-1, out_dim))  

    
    def tx_inputs(a):  return scaler_x.transform(a.reshape(-1, in_dim)).reshape(a.shape)
    def tx_targets(a): return scaler_y.transform(a[..., 1:3].reshape(-1, out_dim)).reshape(a.shape[0], a.shape[1], out_dim)

    X_train_s, X_val_s, X_test_s = tx_inputs(X_train), tx_inputs(X_val), tx_inputs(X_test)
    Y_train_s, Y_val_s, Y_test_s = tx_targets(Y_train), tx_targets(Y_val), tx_targets(Y_test)

    
    model = build_model(seq_len=args.seq_len, in_dim=in_dim, d_model=128, ff_dim=256, num_heads=1)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15, restore_best_weights=True)
    ]
    history = model.fit(
        X_train_s, Y_train_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=args.epochs, batch_size=args.batch, verbose=2, callbacks=callbacks
    )

    
    Y_pred_s = model.predict(X_test_s, verbose=0)         
    
    Y_pred = scaler_y.inverse_transform(Y_pred_s.reshape(-1, out_dim)).reshape(Y_pred_s.shape)
    Y_true = scaler_y.inverse_transform(Y_test_s.reshape(-1, out_dim)).reshape(Y_test_s.shape)

    mse = mse_xy(Y_true, Y_pred)
    curv = mean_curvature_xy(Y_pred)

    print(f"[1-head PE] Test MSE: {mse:.6f} | Curvature: {curv:.6f}")

    model_path = os.path.join(args.models_dir, "transformer_1head.keras")
    xsc_path  = os.path.join(args.models_dir, "transformer_1head_x_scaler.pkl")
    ysc_path  = os.path.join(args.models_dir, "transformer_1head_y_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler_x, xsc_path)
    joblib.dump(scaler_y, ysc_path)

    metrics = {
        "ablation": "heads_1_vs_4",
        "model_path": model_path,
        "x_scaler": xsc_path,
        "y_scaler": ysc_path,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch": args.batch,
        "mse": round(mse, 8),
        "curvature": round(curv, 8),
    }
    with open(os.path.join(args.results_dir, "ablation_heads_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] Wrote {model_path}, {xsc_path}, {ysc_path} and results/ablation_heads_metrics.json")

if __name__ == "__main__":
    main()
