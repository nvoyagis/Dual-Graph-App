# app/main.py
import logging, traceback
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless rendering on Render

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.models import SimRequest, SimResponse
from app.algo import simulate_dual  # expects: (N, seed, tickers, d1, d2, d3, d4)

logger = logging.getLogger("tmfg")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TMFG Simulator API")

# 🔑 CORS: replace <your-firebase-id> with your actual project id
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://algotrades-visualizer.web.app",
        "https://algotrades-visualizer.firebaseapp.com",    ],
    # Allow Cloud Workstations preview during dev (matches *.cloudworkstations.dev)
    allow_origin_regex=r"^https:\/\/.*\.cloudworkstations\.dev$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    # makes the bare URL useful
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/simulate", response_model=SimResponse)
async def simulate(info: SimRequest, request: Request):
    """
    Accepts either `stocks` or `tickers` in the payload (model can normalize either).
    Returns base64 PNGs and a summary. If anything fails, returns a clear error.
    """
    try:
        # Accept both names gracefully even if the model only defines one
        raw_list = (
            getattr(info, "stocks", None)
            or getattr(info, "tickers", None)
            or []
        )
        tickers: List[str] = [t.strip().upper() for t in raw_list if t and t.strip()]
        if not tickers:
            raise ValueError("No stocks provided. Send a non-empty array to `stocks` (or `tickers`).")

        logger.info("simulate input: tickers=%s, date1=%s, date2=%s, date3=%s, date4=%s",
                    tickers[:5] + (["…"] if len(tickers) > 5 else []),
                    getattr(info, "date1", None),
                    getattr(info, "date2", None),
                    getattr(info, "date3", None),
                    getattr(info, "date4", None))

        # 🔑 your algorithm call (adjust N/seed if needed)
        message, summary, tmfg_b64, dual_b64 = simulate_dual(
            100, 1, tickers, info.date1, info.date2, info.date3, info.date4
        )

        return SimResponse(
            message=message or "Simulation complete",
            summary=summary,
            tmfg_png_b64=tmfg_b64,
            dual_png_b64=dual_b64,
        )

    except FileNotFoundError as e:
        logger.error("FileNotFoundError: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"FileNotFoundError: {e}")

    except ValueError as e:
        logger.error("ValueError: %s\n%s", e, traceback.format_exc())
        # 400 for bad input
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Unhandled %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
