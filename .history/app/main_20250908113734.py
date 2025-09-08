# app/main.py
import logging, traceback, base64
from typing import List
import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.models import SimRequest, SimResponse
from app.algo import simulate_dual  # returns: (message, summary, tmfg_b64, dual_b64)

logger = logging.getLogger("tmfg")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TMFG Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://algotrades-visualizer.web.app",
        "https://algotrades-visualizer.firebaseapp.com",
        "https://studio--algotrades-visualizer.us-central1.hosted.app",
    ],
    allow_origin_regex=r"^https:\/\/.*\.cloudworkstations\.dev$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/simulate", response_model=SimResponse)
async def simulate(info: SimRequest, request: Request, debug: int = 0):
    try:
        # debug fast-path to prove pipeline
        if debug == 1:
            tiny = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABt8WQ2QAAAABJRU5ErkJggg=="
            return SimResponse(message="ok", summary={"debug": True},
                               tmfg_png_b64=tiny, dual_png_b64=tiny)

        raw_list = getattr(info, "stocks", None) or getattr(info, "tickers", None) or []
        tickers: List[str] = [t.strip().upper() for t in raw_list if t and t.strip()]
        if not tickers:
            raise HTTPException(status_code=400, detail="No stocks provided. Send array to `stocks` (or `tickers`).")

        logger.info("simulate input: %s ...", tickers[:5] + (["…"] if len(tickers) > 5 else []))

        message, summary, tmfg_b64, dual_b64 = simulate_dual(
            100, 1, tickers, info.date1, info.date2, info.date3, info.date4
        )

        # sanity checks so we never return None
        for name, s in (("tmfg", tmfg_b64), ("dual", dual_b64)):
            if not isinstance(s, str) or len(s) < 50:
                raise ValueError(f"{name} image missing/too short")
            base64.b64decode(s, validate=True)

        return SimResponse(
            message=message or "Simulation complete",
            summary=summary or {},
            tmfg_png_b64=tmfg_b64,
            dual_png_b64=dual_b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("simulate crashed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
