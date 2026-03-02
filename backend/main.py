import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- 1. Paths & Configuration ---
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "19-02-2025-to-19-02-2026.csv"
DATA_PATH = BASE_DIR / "data" / "processed" / "input.txt"
SAVE_PATH = BASE_DIR / "model.pt"

# Hyperparameters
batch_size, block_size = 32, 64 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd, n_head, n_layer, dropout = 128, 4, 4, 0.2

# --- 2. Tokenizer Setup ---
if DATA_PATH.exists() and os.path.getsize(DATA_PATH) > 0:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s if c in stoi] 
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    chars, vocab_size = [], 0

# --- 3. Mini-GPT Architecture ---
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_out, _ = self.sa(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, v_size=None):
        super().__init__()
        v_size = v_size if v_size else vocab_size
        self.token_embedding = nn.Embedding(v_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, v_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        idx = idx[:, -block_size:]
        T = idx.shape[1]
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=device))
        x = self.blocks(x)
        current_vocab_size = self.token_embedding.num_embeddings
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, current_vocab_size), targets.view(-1))
        return logits, loss

    def generate_smart(self, idx, max_new_tokens, temp=1.1, top_k=5):
        self.eval()
        total_conf = 0
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temp
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            total_conf += torch.gather(probs, 1, idx_next).item()
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, (total_conf / max_new_tokens) * 100

# --- 4. FastAPI App & Middleware ---
app = FastAPI(title="MarketGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MOUNT STATIC FILES: Fixes the 404 error for .png files
# This makes images available at http://127.0.0.1:8000/market_trend.png
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "online", "model": "Mini-GPT", "vocab_size": vocab_size}

@app.get("/analysis-summary")
async def get_analysis():
    if not RAW_CSV.exists():
        raise HTTPException(status_code=404, detail="Raw data not found")
    
    try:
        df = pd.read_csv(RAW_CSV)
        df.columns = df.columns.str.strip()
        # Calculate real stats for frontend cards
        avg_trades = df['Total no. of Trades'].mean()
        # Mocking correlation for demo, replace with real logic if needed
        correlation = 0.94 

        return {
            "avgTrades": f"{avg_trades:,.0f}",
            "avgValue": "HIGH_VOL",
            "correlation": correlation,
            "totalDays": len(df),
            "charts": [
                {"title": "MARKET_TREND", "url": "/static/market_trend.png"},
                {"title": "CORRELATION_MAP", "url": "/static/correlation_heatmap.png"}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not SAVE_PATH.exists():
        raise HTTPException(status_code=500, detail="Model weights not found. Train first.")
    
    model = MiniGPT(vocab_size).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    
    input_ids = torch.tensor([encode(request.text)], dtype=torch.long, device=device)
    generated_ids, confidence = model.generate_smart(input_ids, max_new_tokens=40)
    
    return {
        "forecast": decode(generated_ids[0].tolist()),
        "confidence": f"{confidence:.2f}%"
    }