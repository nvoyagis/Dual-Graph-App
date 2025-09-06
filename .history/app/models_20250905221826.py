from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SimRequest(BaseModel):
    stocks: List[str]
    date1: Optional[str] = Field(None, description="YYYY-MM-DD")
    date2: Optional[str] = Field(None, description="YYYY-MM-DD")
    date3: Optional[str] = Field(None, description="YYYY-MM-DD")
    date4: Optional[str] = Field(None, description="YYYY-MM-DD")

class SimResponse(BaseModel):
    message: str
    summary: Dict[str, Any]
    tmfg_png_b64: str     # base64 PNG of TMFG
    dual_png_b64: str     # base64 PNG of dual TMFG
