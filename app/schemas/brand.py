from pydantic import BaseModel
from typing import List, Dict, Optional

class BrandRequest(BaseModel):
    brand: str
    keywords: List[str]
    country: Optional[str] = None
    providers: Optional[List[str]] = ["openai"]

class BrandResponse(BaseModel):
    brand: str
    results: Dict[str, Dict[str, Dict[str, float]]]
