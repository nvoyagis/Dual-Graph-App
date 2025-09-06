from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import SimRequest, SimResponse
from app.algo import simulate

app = FastAPI(title="TMFG Simulator API")

# Allow your Firestudio site to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://algotrades-visualizer.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/simulate", response_model=SimResponse)
def simulate(payload: SimRequest):
    tmfg_b64, dual_b64, summary = simulate(
        tickers=[t.strip().upper() for t in payload.tickers if t.strip()],
        start=payload.start, end=payload.end, n_portfolios=payload.n_portfolios
    )
    return SimResponse(
        message="Simulation complete",
        summary=summary,
        tmfg_png_b64=tmfg_b64,
        dual_png_b64=dual_b64,
    )
