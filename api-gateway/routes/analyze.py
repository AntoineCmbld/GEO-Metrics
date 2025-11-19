import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import httpx
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8081")
REPORT_SERVICE_URL = os.getenv("REPORT_SERVICE_URL", "http://report-service:8080")

router = APIRouter(prefix="/analyze", tags=["analysis"])

class AnalyzeRequest(BaseModel):
    brand_name: str = Field(..., description="Brand name to analyze")
    models: List[str] = Field(..., description="LLM models to query (e.g., ['openai:gpt-4', 'groq:llama-3.1-8b-instant'])")
    keywords: List[str] = Field(..., description="Keywords to search for")
    prompt_template: str = Field(
        default="What do you know about {brand_name}? Describe their products, services, and reputation.",
        description="Prompt template with {brand_name} placeholder"
    )

class AnalyzeResponse(BaseModel):
    report_id: int
    brand_name: str
    overall_score: float
    provider_analyses: List[Dict]
    timestamp: str

@router.post("", response_model=AnalyzeResponse)
async def analyze_brand(request: AnalyzeRequest):
    """
    Orchestrates the full GEO analysis workflow:
    1. Query multiple LLM providers
    2. Analyze keyword presence
    3. Generate report with scores
    """
    try:
        logger.info(f"Starting analysis for brand: {request.brand_name}")
        
        # Step 1: Query all LLM models
        llm_responses = []
        prompt = request.prompt_template.format(brand_name=request.brand_name)
        
        logger.info(f"Querying {len(request.models)} models")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for model_spec in request.models:
                # Parse "provider:model" format
                if ":" in model_spec:
                    provider, model = model_spec.split(":", 1)
                else:
                    provider = model_spec
                    model = None
                
                try:
                    logger.info(f"Querying {provider} with model {model}")
                    
                    # Build request payload
                    payload = {"provider": provider, "prompt": prompt}
                    if model:
                        payload["model"] = model
                    
                    # Query LLM service
                    response = await client.post(
                        f"{LLM_SERVICE_URL}/api/query",
                        json=payload
                    )
                    response.raise_for_status()
                    llm_data = response.json()
                    
                    logger.info(f"Successfully queried {provider}")
                    
                    llm_responses.append({
                        "provider": provider,
                        "model": llm_data.get("model", model or "unknown"),
                        "response": llm_data.get("response", "")
                    })
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error querying {provider}: {e.response.status_code} - {e.response.text}")
                    llm_responses.append({
                        "provider": provider,
                        "model": model or "unknown",
                        "response": f"Error: HTTP {e.response.status_code}"
                    })
                except Exception as e:
                    logger.error(f"Error querying {provider}: {str(e)}", exc_info=True)
                    llm_responses.append({
                        "provider": provider,
                        "model": model or "unknown",
                        "response": f"Error: {str(e)}"
                    })
        
        if not llm_responses:
            raise HTTPException(status_code=500, detail="Failed to get any LLM responses")
        
        logger.info(f"Got {len(llm_responses)} LLM responses, sending to report service")
        
        # Step 2: Send to report service for analysis
        async with httpx.AsyncClient(timeout=30.0) as client:
            report_payload = {
                "brand_name": request.brand_name,
                "keywords": request.keywords,
                "llm_responses": llm_responses
            }
            
            logger.info(f"Sending to report service: {report_payload}")
            
            report_response = await client.post(
                f"{REPORT_SERVICE_URL}/api/reports",
                json=report_payload
            )
            report_response.raise_for_status()
            report_data = report_response.json()
        
        logger.info(f"Successfully created report {report_data.get('id')}")
        
        return AnalyzeResponse(
            report_id=report_data["id"],
            brand_name=report_data["brand_name"],
            overall_score=report_data["overall_score"],
            provider_analyses=report_data["provider_analyses"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in analysis: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code, detail=f"Service error: {e.response.text}")
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")