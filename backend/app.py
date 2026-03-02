from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os

app = FastAPI()

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust Path Handling
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "19-02-2025-to-19-02-2026.csv"

# Serve static images (market_trend.png, etc.)
# This allows the frontend to access images at http://localhost:8000/static/filename.png
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

@app.get("/analysis-summary")
async def get_analysis_summary():
    if not RAW_CSV.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    df = pd.read_csv(RAW_CSV)
    df.columns = df.columns.str.strip()
    
    return {
        "avgTrades": f"{df['Total no. of Trades'].mean():,.0f}",
        "avgValue": f"{df['Traded Value (Rs. in Cr.)'].mean():,.2f}",
        "correlation": round(df['Total no. of Trades'].corr(df['Traded Value (Rs. in Cr.)']), 2),
        "totalDays": len(df),
        "charts": [
            {"title": "Volume Trend Analysis", "url": "http://127.0.0.1:8000/static/market_trend.png"},
            {"title": "Variable Correlation Heatmap", "url": "http://127.0.0.1:8000/static/correlation_heatmap.png"}
        ]
    }

# Ensure your /predict endpoint also returns 'confidence'
@app.post("/predict")
async def predict(data: dict):
    # (Your existing prediction logic from main.py)
    # prediction, confidence = predict_next_outcome(data['text'])
    return {"output": "...", "confidence": 85.5}