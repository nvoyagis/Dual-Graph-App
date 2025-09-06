from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SimRequest(BaseModel):
    tickers: List[str]
    start: Optional[str] = Field(None, description="YYYY-MM-DD")
    end: Optional[str] = Field(None, description="YYYY-MM-DD")
    n_portfolios: int = 50

class SimResponse(BaseModel):
    message: str
    summary: Dict[str, Any]
    tmfg_png_b64: str     # base64 PNG of TMFG
    dual_png_b64: str     # base64 PNG of dual TMFG
