"""
fast_api.py
Robust FastAPI server for AQI anomaly detection.

Usage:
    pipenv run uvicorn fast_api:app --reload

Features:
- Loads processed data and trained artifacts (scaler, iso model, feature list) from ./output/
- /health -> status
- /anomalies -> returns anomaly records (requires processed_with_flags.csv)
- /predict -> accepts flexible JSON payload and returns is_anomaly (fills missing features with 0.0 and warns)
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Extra

# -----------------------
# Configuration / paths
# -----------------------
OUTPUT_DIR = "./output"
PROCESSED = os.path.join(OUTPUT_DIR, "processed_with_flags.csv")
ISO_FEATURES_PATH = os.path.join(OUTPUT_DIR, "iso_feature_list.joblib")
SCALER_PATH = os.path.join(OUTPUT_DIR, "iso_scaler.joblib")
ISO_PATH = os.path.join(OUTPUT_DIR, "iso_model.joblib")

# -----------------------
# App & artifact loading
# -----------------------
app = FastAPI(title="AQI Anomaly API")

# load processed dataframe if present
_PROCESSED_DF = None
if os.path.exists(PROCESSED):
    try:
        _PROCESSED_DF = pd.read_csv(PROCESSED, parse_dates=["timestamp"]).set_index("timestamp")
    except Exception:
        _PROCESSED_DF = pd.read_csv(PROCESSED, parse_dates=["timestamp"], infer_datetime_format=True)
else:
    _PROCESSED_DF = None

# load models & feature list (if available)
_iso_features = joblib.load(ISO_FEATURES_PATH) if os.path.exists(ISO_FEATURES_PATH) else None
_scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
_iso = joblib.load(ISO_PATH) if os.path.exists(ISO_PATH) else None

# -----------------------
# Pydantic model
# -----------------------
class PredictRow(BaseModel):
    # common pollutant & weather fields (all optional)
    PM25: float = None
    PM2_5: float = None
    PM2dot5: float = None
    PM2dot5_alt: float = None
    PM10: float = None
    NO2: float = None
    SO2: float = None
    CO: float = None
    O3: float = None
    Temperature: float = None
    Humidity: float = None
    Wind_Speed: float = None

    class Config:
        extra = Extra.allow  # accept additional dynamic keys

# -----------------------
# Utility helpers
# -----------------------
def _candidate_keys_for(feature_name: str):
    """
    Given a canonical feature name (e.g., 'PM2.5'), return likely keys to check in incoming payload.
    """
    k = feature_name
    cands = [k, k.replace(".", ""), k.replace(".", "_"), k.replace(" ", "_"), k.replace(" ", "")]
    cands += [s.lower() for s in list(cands)]
    # also include common variants
    if k.lower().startswith("pm2"):
        cands += ["pm25", "pm2_5", "pm2dot5"]
    return list(dict.fromkeys(cands))  # unique preserving order

def _extract_value_from_payload(payload: dict, feature_name: str):
    """
    Try to extract a numeric value for feature_name from payload using a set of candidate keys.
    Returns (value_or_None, matched_key_or_None).
    """
    for cand in _candidate_keys_for(feature_name):
        if cand in payload and payload[cand] is not None:
            return payload[cand], cand
    # case-insensitive match fallback
    for k, v in payload.items():
        if v is None:
            continue
        if k.lower().replace(".", "").replace(" ", "").replace("_", "") == feature_name.lower().replace(".", "").replace(" ", "").replace("_", ""):
            return v, k
    return None, None

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "processed_exists": os.path.exists(PROCESSED),
        "model_loaded": _iso is not None and _scaler is not None and _iso_features is not None,
        "feature_count_expected": len(_iso_features) if _iso_features is not None else None
    }

@app.get("/anomalies")
def anomalies(limit: int = 200):
    if _PROCESSED_DF is None:
        raise HTTPException(status_code=404, detail="Processed data not found. Run aqi_models.py to generate processed_with_flags.csv.")
    if "anom_any" not in _PROCESSED_DF.columns:
        return {"count": 0, "records": []}
    anoms = _PROCESSED_DF[_PROCESSED_DF["anom_any"] == True].tail(limit)
    return {"count": len(anoms), "records": anoms.reset_index().to_dict(orient="records")}

@app.post("/predict")
async def predict(req: Request):
    """
    Accepts flexible JSON payload. Builds feature vector according to saved iso feature list.
    Missing features are filled with 0.0 (and returned in warning).
    """
    if _scaler is None or _iso is None or _iso_features is None:
        raise HTTPException(status_code=500, detail="Model artifacts not available. Run aqi_models.py first to create scaler, model, and feature list.")

    try:
        payload = await req.json()
        # payload may contain keys not declared in PredictRow; that's okay
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    X_list = []
    missing = []
    matched = {}

    for feat in _iso_features:
        val, matched_key = _extract_value_from_payload(payload, feat)
        if val is None:
            # fallback to 0.0
            val = 0.0
            missing.append(feat)
        else:
            matched[feat] = matched_key
        # ensure numeric
        try:
            X_list.append(float(val))
        except Exception:
            X_list.append(0.0)
            missing.append(feat)

    # Build array and check shape
    X = np.array(X_list).reshape(1, -1)

    # Transform and predict
    try:
        Xs = _scaler.transform(X)
    except Exception as e:
        # return actionable debug info
        expected = getattr(_scaler, "n_features_in_", None)
        raise HTTPException(status_code=400, detail=f"Scaler transform failed: {e}. Expected features: {expected}, provided: {len(X_list)}.")

    pred = _iso.predict(Xs)[0]
    response = {"is_anomaly": bool(pred == -1)}
    if missing:
        response["warning_missing_features_filled_with_0"] = missing
    if matched:
        response["matched_input_keys"] = matched
    return response
