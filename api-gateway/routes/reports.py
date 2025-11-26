import os
from fastapi import APIRouter, Request, Depends
from typing import List
from pydantic import BaseModel, Field
from .proxy import proxy_request
from .auth_utils import get_user_id

router = APIRouter(prefix="/reports", tags=["reports"])

REPORT_SERVICE_URL = os.getenv("REPORT_SERVICE_URL", "http://report-service:8080")

# Define AnalyzeRequest and AnalyzeResponse for docs
class AnalyzeRequest(BaseModel):
    brand_name: str = Field(..., description="Brand name to analyze")
    models: List[str] = Field(..., description="LLM models to query")
    keywords: List[str] = Field(..., description="Keywords to query")
    regions: List[str] = Field(..., description="Region contexts")
    languages: List[str] = Field(..., description="Language codes (use 'default' for original prompts, or language codes like 'es', 'fr')")
    prompt_templates: List[str] = Field(..., description="Prompt templates with {keyword} placeholder")

class AnalyzeResponse(BaseModel):
    report_id: int
    brand_name: str
    timestamp: str
    total_queries: int
    successful_queries: int
    failed_queries: int

@router.get("/health")
async def report_health(request: Request, user_id: int = Depends(get_user_id)):
    """Check report service health"""
    target_url = f"{REPORT_SERVICE_URL}/api/health"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)

@router.get("")
async def list_reports(request: Request, user_id: int = Depends(get_user_id)):
    """List all reports"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)

@router.get("/{report_id}")
async def get_report(report_id: int, request: Request, user_id: int = Depends(get_user_id)):
    """Get a specific report"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)

@router.delete("/{report_id}")
async def delete_report(report_id: int, request: Request, user_id: int = Depends(get_user_id)):
    """Delete a report"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)

@router.get("/{report_id}/llm-responses")
async def get_report_llm_responses(report_id: int, request: Request, user_id: int = Depends(get_user_id)):
    """Get report LLM responses"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}/llm-responses"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_brand(
    request: Request,
    analyze_request: AnalyzeRequest,
    user_id: int = Depends(get_user_id)
):
    """Analyze brand visibility across LLM providers (proxied to report-service)"""
    target_url = f"{REPORT_SERVICE_URL}/api/analyze"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(
        request,
        target_url,
        extra_headers=headers,
        json=analyze_request.model_dump()
    )