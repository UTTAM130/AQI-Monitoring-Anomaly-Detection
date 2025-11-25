"""
aqi_models.py
End-to-end anomaly detection model builder
Compatible with your dataset: global_air_quality_data_10000.csv (daily, wide format)
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSED_IN = os.path.join(OUTPUT_DIR, "processed_features.csv")
PROCESSED_OUT = os.path.join(OUTPUT_DIR, "processed_with_flags.csv")

SCALER_PATH = os.path.join(OUTPUT_DIR, "iso_scaler.joblib")
ISO_PATH = os.path.join(OUTPUT_DIR, "iso_model.joblib")
FEATURE_LIST_PATH = os.path.join(OUTPUT_DIR, "iso_feature_list.joblib")

LSTM_MODEL_PATH = os.path.join(OUTPUT_DIR, "lstm_autoencoder.h5")
LSTM_SCALER_PATH = os.path.join(OUTPUT_DIR, "lstm_minmax_scaler.joblib")

# ----------------------------
# Load processed dataset
# ----------------------------

def load_processed(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


# ----------------------------
# Z-SCORE ANOMALY DETECTION
# ----------------------------

def compute_zscore(df, col="PM2.5"):
    roll_mean = df[col].rolling(window=7, min_periods=1).mean()
    roll_std = df[col].rolling(window=7, min_periods=1).std().fillna(0)
    df["zscore"] = (df[col] - roll_mean) / (roll_std + 1e-9)
    df["anom_z"] = df["zscore"].abs() > 3
    return df


# ----------------------------
# ISOLATION FOREST
# ----------------------------

def train_iso(df, features):

    X = df[features].fillna(0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )
    iso.fit(Xs)

    pred = iso.predict(Xs)
    df["anom_iso"] = pred == -1

    return df, scaler, iso


# ----------------------------
# OPTIONAL: LSTM AUTOENCODER
# ----------------------------

def train_lstm(df, seq_features, seq_len=7, epochs=20):

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except:
        print("[LSTM] TensorFlow not installed. Skipping LSTM model.")
        df["anom_lstm"] = False
        return df, None, None

    mms = MinMaxScaler()
    X = df[seq_features].fillna(0).values
    X_scaled = mms.fit_transform(X)

    sequences = []
    for i in range(len(X_scaled) - seq_len):
        sequences.append(X_scaled[i:i+seq_len])
    sequences = np.array(sequences)

    if len(sequences) < 10:
        print("[LSTM] Not enough sequences for training. Skipping.")
        df["anom_lstm"] = False
        return df, None, None

    n_features = sequences.shape[2]

    # Build autoencoder
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)

    ae = models.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    ae.fit(sequences, sequences, epochs=epochs, batch_size=16, verbose=1)

    recon = ae.predict(sequences)
    mse = np.mean(np.mean((recon - sequences)**2, axis=2), axis=1)

    mse_series = pd.Series(mse, index=df.index[seq_len:])
    threshold = mse_series.mean() + 3*mse_series.std()

    df["anom_lstm"] = False
    df.loc[mse_series[mse_series > threshold].index, "anom_lstm"] = True

    return df, mms, ae


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def main():

    print("=== Starting AQI Model Pipeline ===")

    df = load_processed(PROCESSED_IN)
    print("Loaded:", df.shape)

    # Ensure PM2.5 exists
    if "PM2.5" not in df.columns:
        raise RuntimeError("Processed dataset missing 'PM2.5'")

    # STEP 1 — Z-SCORE
    df = compute_zscore(df, "PM2.5")
    print("[Z-SCORE] Done. Z anomalies:", df["anom_z"].sum())

    # STEP 2 — ISOLATION FOREST FEATURE LIST
    features_iso = []

    for col in ["PM2.5","PM10","NO2","SO2","CO","O3",
                "Temperature","Humidity","Wind Speed",
                "roll_mean_7d","roll_std_7d"]:
        if col in df.columns:
            features_iso.append(col)

    print("[ISO] Features:", features_iso)

    # Save feature list
    joblib.dump(features_iso, FEATURE_LIST_PATH)
    print("[ISO] Saved feature list:", FEATURE_LIST_PATH)

    # Train ISO
    df, scaler, iso = train_iso(df, features_iso)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(iso, ISO_PATH)

    print("[ISO] Model saved.")

    # STEP 3 — LSTM (Optional)
    lstm_features = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df.columns]

    df, mms, lstm_model = train_lstm(df, lstm_features)

    if mms is not None:
        joblib.dump(mms, LSTM_SCALER_PATH)
        lstm_model.save(LSTM_MODEL_PATH)

    # Combine anomaly votes
    df["anom_votes"] = df[["anom_z","anom_iso","anom_lstm"]].sum(axis=1)
    df["anom_any"] = df["anom_votes"] > 0

    df.to_csv(PROCESSED_OUT)
    print("Saved:", PROCESSED_OUT)


if __name__ == "__main__":
    main()
