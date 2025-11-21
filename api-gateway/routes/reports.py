import os
from fastapi import APIRouter, Request
from .proxy import proxy_request

router = APIRouter(prefix="/reports", tags=["reports"])

REPORT_SERVICE_URL = os.getenv("REPORT_SERVICE_URL", "http://report-service:8080")

@router.get("/health")
async def report_health(request: Request):
    """Check report service health"""
    target_url = f"{REPORT_SERVICE_URL}/api/health"
    return await proxy_request(request, target_url)

@router.post("")
async def create_report(request: Request):
    """Create a new report"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports"
    return await proxy_request(request, target_url)

@router.get("")
async def list_reports(request: Request):
    """List all reports"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports"
    return await proxy_request(request, target_url)

@router.get("/{report_id}")
async def get_report(report_id: int, request: Request):
    """Get a specific report"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}"
    return await proxy_request(request, target_url)

@router.delete("/{report_id}")
async def delete_report(report_id: int, request: Request):
    """Delete a report"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}"
    return await proxy_request(request, target_url)

@router.get("/{report_id}/score")
async def get_report_score(report_id: int, request: Request):
    """Get report score"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}/score"
    return await proxy_request(request, target_url)

@router.get("/{report_id}/llm-responses")
async def get_report_llm_responses(report_id: int, request: Request):
    """Get report LLM responses"""
    target_url = f"{REPORT_SERVICE_URL}/api/reports/{report_id}/llm-responses"
    return await proxy_request(request, target_url)