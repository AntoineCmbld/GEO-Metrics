from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import re

app = FastAPI(title="GEO Metrics")

# Initialize OpenAI client (make sure you set OPENAI_API_KEY in your environment)
client = OpenAI()

# --- Data models ---
class BrandRequest(BaseModel):
    brand: str
    keywords: List[str]

class BrandResponse(BaseModel):
    brand: str
    results: Dict[str, Dict[str, float]]  # {keyword: {"mention_count": int, "mention_ratio": float}}

# --- Helper function ---
def count_brand_mentions(text: str, brand: str) -> int:
    """Counts how many times the brand name appears (case-insensitive)."""
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    return len(pattern.findall(text))

# --- Main route ---
@app.post("/analyze", response_model=BrandResponse)
async def analyze_brand_mentions(request: BrandRequest):
    brand = request.brand.strip()
    if not brand:
        raise HTTPException(status_code=400, detail="Brand name cannot be empty.")

    results = {}

    for keyword in request.keywords:
        try:
            # Ask GPT about the keyword
            prompt = (
                f"Let's talk about '{keyword}'. "
                f"What are the most well-known brands or companies associated with this topic?"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

            answer = response.choices[0].message.content
            mention_count = count_brand_mentions(answer, brand)
            mention_ratio = mention_count / max(1, len(answer.split()))  # basic normalization

            results[keyword] = {
                "mention_count": mention_count,
                "mention_ratio": round(mention_ratio, 4)
            }

        except Exception as e:
            results[keyword] = {"error": str(e)}

    return {"brand": brand, "results": results}
