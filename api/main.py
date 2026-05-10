"""
Fraud Detection API — Phase 3: Backend API Engineering
=======================================================
Day 9  : FastAPI scaffold + health check endpoint
Day 10 : Pydantic schema + model/scaler lifespan loading
Day 11 : POST /predict endpoint with full integration test support
"""

from contextlib import asynccontextmanager
import warnings
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ─── Day 10: Pydantic Schema ────────────────────────────────────────────────
# Strictly validates the 30 raw input features required by the model.
# Time and Amount are raw values — the API applies RobustScaler internally.
class TransactionData(BaseModel):
    Time: float
    Amount: float
    V1:  float
    V2:  float
    V3:  float
    V4:  float
    V5:  float
    V6:  float
    V7:  float
    V8:  float
    V9:  float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# ─── Day 10: Lifespan — Load Model & Scaler Once at Startup ─────────────────
# Globals to hold loaded artifacts
ml_model = None
scaler   = None

# Paths are relative to the api/ directory at runtime
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, scaler
    try:
        ml_model = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        print("✅ Model and scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠️  Artifact not found: {e}")
        print("   Run 02_model_training.ipynb to generate model.pkl and scaler.pkl first.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
    yield
    # Cleanup on shutdown
    ml_model = None
    scaler   = None


# ─── Day 9: FastAPI Initialisation ──────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection engine. "
        "Send a transaction's 30 raw features and receive a binary fraud "
        "prediction alongside a confidence probability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Day 9: Health Check Endpoint ───────────────────────────────────────────
@app.get("/", tags=["Health"])
def read_root():
    """Returns API operational status."""
    return {"status": "API is operational"}


@app.get("/health", tags=["Health"])
def health_check():
    """Returns model readiness alongside API status."""
    model_ready = ml_model is not None and scaler is not None
    return {
        "status": "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
    }


# ─── Day 11: POST /predict Endpoint ─────────────────────────────────────────
@app.post("/predict", tags=["Inference"])
def predict_fraud(transaction: TransactionData):
    """
    Accepts a raw credit card transaction, applies RobustScaling to
    Time and Amount, and returns a fraud prediction with confidence score.
    """
    if ml_model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure model.pkl and scaler.pkl exist in models/.",
        )

    # Convert Pydantic model → dict
    data = transaction.model_dump()

    # Apply the same RobustScaler that was used during training
    # Scaler was fitted separately on Amount and Time; we reuse it here.
    scaled_amount = scaler.transform([[data["Amount"]]])[0][0]
    scaled_time   = scaler.transform([[data["Time"]]])[0][0]

    # Build feature DataFrame in the exact column order used during training:
    # [scaled_amount, scaled_time, V1 … V28]
    feature_cols = (
        ["scaled_amount", "scaled_time"]
        + [f"V{i}" for i in range(1, 29)]
    )

    row = (
        [scaled_amount, scaled_time]
        + [data[f"V{i}"] for i in range(1, 29)]
    )

    input_df = pd.DataFrame([row], columns=feature_cols)

    # Inference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction  = int(ml_model.predict(input_df)[0])
        probability = float(ml_model.predict_proba(input_df)[0][1])

    return {
        "fraud_prediction":       prediction,
        "confidence_probability": round(probability, 6),
        "status":                 "Fraudulent" if prediction == 1 else "Legitimate",
    }


# ─── Day 9: Uvicorn Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
