import os
import logging
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8081")
REPORT_SERVICE_URL = os.getenv("REPORT_SERVICE_URL", "http://report-service:8080")

router = APIRouter(prefix="/analyze", tags=["analysis"])

class AnalyzeRequest(BaseModel):
    brand_name: str = Field(..., description="Brand name to analyze")
    models: List[str] = Field(..., description="LLM models to query (e.g., ['meta-llama/Llama-3.1-8B-Instruct'])")
    keywords: List[str] = Field(..., description="Keywords to query separately")
    prompt_template: str = Field(
        default="What do you know about {keyword}? What brands come to your mind when you think of {keyword}?",
        description="Prompt template with {keyword} placeholder"
    )

class AnalyzeResponse(BaseModel):
    report_id: int
    brand_name: str
    overall_score: float
    provider_analyses: List[Dict]
    timestamp: str

@router.post("", response_model=AnalyzeResponse)
async def analyze_brand(request: AnalyzeRequest):
    """Analyze brand visibility across LLM providers"""
    logger.info(f"Starting analysis for brand: {request.brand_name}")
    
    provider_analyses = []
    all_llm_responses = []
    has_errors = False
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Process each keyword
        for keyword in request.keywords:
            keyword_analysis = {
                "keyword": keyword,
                "provider_results": [],
                "total_mentions": 0,
                "visibility_score": 0.0
            }
            
            # Generate prompt for this keyword
            prompt = request.prompt_template.format(keyword=keyword)
            
            # Query each model
            for model in request.models:
                try:
                    # Query LLM service
                    llm_response = await client.post(
                        f"{LLM_SERVICE_URL}/api/query",
                        json={
                            "model": model,
                            "prompt": prompt
                        }
                    )
                    
                    if llm_response.status_code != 200:
                        error_detail = llm_response.text
                        logger.error(f"LLM query failed for {model} - {error_detail}")
                        has_errors = True
                        keyword_analysis["provider_results"].append({
                            "model": model,
                            "response": f"Error: HTTP {llm_response.status_code}",
                            "brand_mentions": 0
                        })
                        continue
                    
                    llm_data = llm_response.json()
                    response_text = llm_data.get("response", "")
                    
                    # Count brand mentions (case-insensitive)
                    brand_mentions = response_text.lower().count(request.brand_name.lower())
                    
                    keyword_analysis["provider_results"].append({
                        "model": model,
                        "response": response_text,
                        "brand_mentions": brand_mentions
                    })
                    
                    keyword_analysis["total_mentions"] += brand_mentions
                    
                    # Store for report creation
                    all_llm_responses.append({
                        "model": model,
                        "keyword": keyword,
                        "response": response_text
                    })
                    
                except Exception as e:
                    logger.error(f"Error querying {model} - {str(e)}")
                    has_errors = True
                    keyword_analysis["provider_results"].append({
                        "model": model,
                        "response": f"Error: {str(e)}",
                        "brand_mentions": 0
                    })
            
            # Calculate visibility score for this keyword (mentions per model queried)
            num_models = len(request.models)
            if num_models > 0:
                keyword_analysis["visibility_score"] = keyword_analysis["total_mentions"] / num_models
            
            provider_analyses.append(keyword_analysis)
    
    # Don't create report if there were errors
    if has_errors:
        logger.error("Analysis failed with errors, not creating report")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Analysis failed",
                "message": "One or more LLM queries failed. Please check the logs and try again.",
                "partial_results": provider_analyses
            }
        )
    
    # Calculate overall score (average visibility across all keywords)
    overall_score = sum(k["visibility_score"] for k in provider_analyses) / len(provider_analyses) if provider_analyses else 0.0
    
    # Create report in report-service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            report_response = await client.post(
                f"{REPORT_SERVICE_URL}/api/reports",
                json={
                    "brand_name": request.brand_name,
                    "keywords": request.keywords,
                    "llm_responses": all_llm_responses
                }
            )
            
            if report_response.status_code != 201:
                raise HTTPException(
                    status_code=report_response.status_code,
                    detail=f"Failed to create report: {report_response.text}"
                )
            
            report_data = report_response.json()
            
            return AnalyzeResponse(
                report_id=report_data["id"],
                brand_name=request.brand_name,
                overall_score=overall_score,
                provider_analyses=provider_analyses,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    except httpx.RequestError as e:
        logger.error(f"Failed to create report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create report: {str(e)}"
        )