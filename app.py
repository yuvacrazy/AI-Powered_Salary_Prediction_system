# app.py
import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "salary_model_lgb.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "encoders_lgb.pkl")
DATA_PATH = os.getenv("DATA_PATH", "clean_salary_dataset.csv")
API_KEY = os.getenv("API_KEY", "NGROK_AUTH_TOKEN")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# -----------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smartpay-api")

# -----------------------------------------------------------
# LOAD MODEL & ENCODERS
# -----------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    logger.info("✅ Model and encoders loaded successfully.")
except Exception as e:
    logger.exception("❌ Failed to load model or encoders: %s", e)
    raise RuntimeError("Model or encoder loading failed.")

# -----------------------------------------------------------
# FASTAPI INITIALIZATION
# -----------------------------------------------------------
app = FastAPI(title="SmartPay Backend API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------
def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return True

# -----------------------------------------------------------
# REQUEST SCHEMA
# -----------------------------------------------------------
class PredictRequest(BaseModel):
    age: int = Field(..., ge=16, le=100)
    education: str
    job_title: str
    hours_per_week: int = Field(..., ge=1, le=100)
    gender: str
    marital_status: str

    @validator('education', 'job_title', 'gender', 'marital_status')
    def validate_nonempty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class PredictResponse(BaseModel):
    predicted_salary_usd: float
    model_version: Optional[str] = "LightGBM v1.0"

# -----------------------------------------------------------
# PREDICT ENDPOINT
# -----------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict_salary(req: PredictRequest, auth: bool = Depends(api_key_auth)):
    try:
        feat_order = ['age', 'education', 'job_title', 'hours_per_week', 'gender', 'marital_status']
        vals = []
        d = req.dict()

        # Encode categorical features
        for f in feat_order:
            if f in ['education', 'job_title', 'gender', 'marital_status']:
                enc = encoders.get(f)
                if not enc:
                    raise HTTPException(status_code=400, detail=f"Encoder missing for {f}")
                try:
                    vals.append(int(enc.transform([str(d[f])])[0]))
                except Exception:
                    vals.append(-1)  # unseen category
            else:
                vals.append(float(d[f]))

        X = np.array([vals])
        pred = model.predict(X)[0]
        return PredictResponse(predicted_salary_usd=float(pred))

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

# -----------------------------------------------------------
# ANALYSIS ENDPOINT
# -----------------------------------------------------------
@app.get("/analyze")
def analyze(auth: bool = Depends(api_key_auth)):
    try:
        df = pd.read_csv(DATA_PATH)
        if "salary_in_usd" not in df.columns:
            raise ValueError("Column 'salary_in_usd' not found in dataset")

        summary = {
            "record_count": len(df),
            "average_salary": round(float(df["salary_in_usd"].mean()), 2),
            "max_salary": round(float(df["salary_in_usd"].max()), 2),
            "min_salary": round(float(df["salary_in_usd"].min()), 2)
        }
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# -----------------------------------------------------------
# EXPLAINABILITY ENDPOINT
# -----------------------------------------------------------
@app.get("/explain")
def explain(auth: bool = Depends(api_key_auth)):
    try:
        importance = model.feature_importances_
        features = model.feature_name_ if hasattr(model, "feature_name_") else [
            'age', 'education', 'job_title', 'hours_per_week', 'gender', 'marital_status'
        ]

        top_idx = np.argsort(importance)[::-1][:5]
        top_features = [
            {"feature": features[i], "importance": round(float(importance[i]), 3)}
            for i in top_idx
        ]
        return {"status": "success", "top_features": top_features}
    except Exception as e:
        logger.exception("Explainability failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Explainability failed: {e}")

# -----------------------------------------------------------
# HEALTH ENDPOINT
# -----------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

# -----------------------------------------------------------
# ROOT ENDPOINT
# -----------------------------------------------------------
@app.get("/")
def root():
    return {"service": "SmartPay Backend API", "status": "running"}

# -----------------------------------------------------------
# RUN SERVER
# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

