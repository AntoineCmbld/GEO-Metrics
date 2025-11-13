from fastapi import APIRouter, HTTPException
from app.schemas.brand import BrandRequest, BrandResponse
from app.services.ai_service import generate_ai_response
from app.utils.text_utils import count_brand_mentions

router = APIRouter()

@router.post("/analyze", response_model=BrandResponse)
async def analyze_brand_mentions(request: BrandRequest):
    brand = request.brand.strip()
    if not brand:
        raise HTTPException(status_code=400, detail="Brand name cannot be empty.")

    valid_providers = {"openai", "google_gemini", "groq"}
    for p in request.providers:
        if p.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {p}")

    results = {}

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
