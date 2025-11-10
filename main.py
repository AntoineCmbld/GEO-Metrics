from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
import google.genai as genai
from groq import Groq
import os
import re

app = FastAPI(title="GEO Metrics Multi-AI")

# --- Load API keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Initialize clients ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
google_client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Pydantic models ---
class BrandRequest(BaseModel):
    brand: str
    keywords: List[str]
    country: Optional[str] = None
    providers: Optional[List[str]] = ["openai"]  # e.g., ["openai","google_gemini"]

class BrandResponse(BaseModel):
    brand: str
    results: Dict[str, Dict[str, Dict[str, float]]]

# --- Helper: count brand mentions ---
def count_brand_mentions(text: str, brand: str) -> int:
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    return len(pattern.findall(text))

# --- Helper: translate prompt using OpenAI only ---
async def translate_prompt(prompt: str, country: str) -> str:
    translation_prompt = f"Translate the following text into the main language spoken in {country}:\n{prompt}"
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": translation_prompt}],
        max_tokens=500
    )
    return resp.choices[0].message.content

# --- Helper: generate AI response ---
async def generate_ai_response(prompt: str, provider: str, country: Optional[str] = None) -> str:
    # Translate using GPT if country is specified
    if country:
        prompt = await translate_prompt(prompt, country)
        prompt += f" in {country}"

    provider = provider.lower()

    if provider == "openai":
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content

    elif provider == "google_gemini":
        resp = google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return resp.text

    elif provider == "groq":
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unsupported provider: {provider}")

# --- Main API route ---
@app.post("/analyze", response_model=BrandResponse)
async def analyze_brand_mentions(request: BrandRequest):
    brand = request.brand.strip()
    if not brand:
        raise HTTPException(status_code=400, detail="Brand name cannot be empty.")

    # Validate providers
    valid_providers = {"openai", "google_gemini", "groq"}
    for p in request.providers:
        if p.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {p}")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for keyword in request.keywords:
        results[keyword] = {}
        for provider in request.providers:
            provider = provider.lower()
            try:
                prompt = (
                    f"Let's talk about '{keyword}'. "
                    "What are the most well-known brands or companies associated with this topic?"
                )

                answer = await generate_ai_response(prompt, provider, request.country)

                mention_count = count_brand_mentions(answer, brand)
                mention_ratio = mention_count / max(1, len(answer.split()))

                results[keyword][provider] = {
                    "mention_count": mention_count,
                    "mention_ratio": round(mention_ratio, 4)
                }

            except Exception as e:
                results[keyword][provider] = {"error": str(e)}

    return {"brand": brand, "results": results}
