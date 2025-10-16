from typing import Any, Dict, Optional
from pydantic import BaseModel

class AnalyzeResult(BaseModel):
    verdict: str
    ai_score: float
    confidence: float
    parts: Dict[str, Any]
    details: Dict[str, Any]

class URLPayload(BaseModel):
    url: str
