from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
import re

app = FastAPI(title="GEO Metrics")

client = OpenAI()

# --- Data models ---
class BrandRequest(BaseModel):
    brand: str
    keywords: List[str]
    country: Optional[str] = None

class BrandResponse(BaseModel):
    brand: str
    results: Dict[str, Dict[str, float]]

# --- Helper function ---
def count_brand_mentions(text: str, brand: str) -> int:
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    return len(pattern.findall(text))

# --- Translation helper ---
async def translate_prompt(prompt: str, country: str) -> str:
    """
    Translate the prompt into the main language of the given country.
    """
    translation_prompt = f"Translate the following text to the main language spoken in {country}:\n{prompt}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": translation_prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# --- Main route ---
@app.post("/analyze", response_model=BrandResponse)
async def analyze_brand_mentions(request: BrandRequest):
    brand = request.brand.strip()
    if not brand:
        raise HTTPException(status_code=400, detail="Brand name cannot be empty.")

    results = {}

    for keyword in request.keywords:
        try:
            prompt = f"Let's talk about '{keyword}'. What are the most well-known brands or companies associated with this topic?"
            
            # Translate prompt if country is provided
            if request.country:
                prompt = await translate_prompt(prompt, request.country)
                # Append country context
                prompt += f" in {request.country}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

            answer = response.choices[0].message.content
            mention_count = count_brand_mentions(answer, brand)
            mention_ratio = mention_count / max(1, len(answer.split()))

            results[keyword] = {
                "mention_count": mention_count,
                "mention_ratio": round(mention_ratio, 4)
            }

        except Exception as e:
            results[keyword] = {"error": str(e)}

    return {"brand": brand, "results": results}
