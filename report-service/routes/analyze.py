import os
import logging
import httpx
import asyncio
from fastapi import APIRouter, Header, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
from database import get_db
from services.report_generator import ReportGenerator
from deep_translator import GoogleTranslator
from routes.reports import LLMResponseData

logger = logging.getLogger(__name__)
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8081")

router = APIRouter(prefix="/analyze", tags=["analysis"])

class AnalyzeRequest(BaseModel):
    brand_name: str = Field(..., description="Brand name to analyze")
    models: List[str] = Field(default=["meta-llama/Llama-3.1-8B-Instruct"], description="LLM models to query")
    keywords: List[str] = Field(..., description="Keywords to query")
    regions: List[str] = Field(default=["Global"], description="Region contexts")
    languages: List[str] = Field(
        default=["default"],
        description="Language codes (use 'default' for original prompts, or language codes like 'es', 'fr')"
    )
    prompt_templates: List[str] = Field(
        default=["What do you know about {keyword}? What brands come to your mind when you think of {keyword}?"],
        description="List of prompt templates with {keyword} placeholder"
    )

class AnalyzeResponse(BaseModel):
    report_id: int
    brand_name: str
    timestamp: str
    total_queries: int
    successful_queries: int
    failed_queries: int

def translate_prompt(prompt: str, target_lang: str) -> str:
    """Translate the complete prompt"""
    if target_lang == "default":
        return prompt
    
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(prompt)
        logger.info(f"Translated to {target_lang}: '{prompt[:30]}...' -> '{translated[:30]}...'")
        return translated
    except Exception as e:
        logger.error(f"Translation to {target_lang} failed: {e}")
        return prompt

async def query_llm(client: httpx.AsyncClient, model: str, prompt: str, region: Optional[str]):
    """Query a single LLM model"""
    try:
        payload = {"model": model, "prompt": prompt}
        if region and region != "Global":
            payload["region"] = region
            
        response = await client.post(f"{LLM_SERVICE_URL}/api/query", json=payload, timeout=60.0)
        
        if response.status_code != 200:
            logger.error(f"LLM query failed: {model} - {response.text}")
            return {"success": False, "model": model, "prompt_text": prompt}
        
        data = response.json()
        return {
            "success": True,
            "model": model,
            "prompt_text": prompt,
            "response": data.get("response", "")
        }
    except Exception as e:
        logger.error(f"Error querying {model}: {e}")
        return {"success": False, "model": model, "prompt_text": prompt}

@router.post("", response_model=AnalyzeResponse)
async def analyze_brand(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    """Analyze brand visibility across LLM providers"""
    logger.info(
        f"Analyzing {request.brand_name}: "
        f"{len(request.models)} models"
        f"{len(request.keywords)} keywords"
        f"{len(request.prompt_templates)} templates"
        f"{len(request.languages)} languages"
        f"{len(request.regions)} regions"
    )
    
    # Step 1: Build jobs
    jobs = []
    for region in request.regions:
        for language in request.languages:
            for prompt_template in request.prompt_templates:
                for keyword in request.keywords:
                    prompt = prompt_template.format(keyword=keyword)
                    translated_prompt = translate_prompt(prompt, language)
                    for model in request.models:
                        jobs.append({
                            "model": model,
                            "keyword": keyword,
                            "language_code": language,
                            "region": region,
                            "prompt_text": translated_prompt,
                            "prompt_template": prompt_template
                        })

    # Step 2: Run queries
    tasks = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for job in jobs:
            tasks.append(query_llm(client, job["model"], job["prompt_text"], job["region"]))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 3: Build LLMResponseData only for successful queries
    llm_responses = []
    for job, result in zip(jobs, results):
        if isinstance(result, dict) and result.get("success"):
            llm_responses.append(
                LLMResponseData(
                    model=job["model"],
                    keyword=job["keyword"],
                    language_code=job["language_code"],
                    region=job["region"],
                    prompt_text=job["prompt_text"],
                    response=result.get("response", "")
                ).model_dump()
            )

    # Step 4: Store/transmit only llm_responses
    generator = ReportGenerator(db)
    report = generator.generate_report(
        brand_name=request.brand_name,
        user_id=x_user_id,
        llm_responses=llm_responses
    )
    
    successful = len(llm_responses)
    failed = len(results) - successful

    return AnalyzeResponse(
        report_id=report.id, # type: ignore
        brand_name=request.brand_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_queries=len(results),
        successful_queries=successful,
        failed_queries=failed
    )