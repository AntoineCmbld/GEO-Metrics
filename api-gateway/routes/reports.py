import os
from fastapi import APIRouter, Request
from .proxy import proxy_request

REPORT_SERVICE_URL = os.getenv("REPORT_SERVICE_URL", "http://report-service:8080")

router = APIRouter(prefix="/reports")

@router.get("/health")
async def report_health(request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, "/api/health")

@router.post("")
async def create_report(request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, "/api/reports")

@router.get("")
async def list_reports(request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, "/api/reports")

@router.get("/{report_id}")
async def get_report(report_id: int, request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, f"/api/reports/{report_id}")

@router.delete("/{report_id}")
async def delete_report(report_id: int, request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, f"/api/reports/{report_id}")

@router.get("/{report_id}/score")
async def get_report_score(report_id: int, request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, f"/api/reports/{report_id}/score")

@router.get("/{report_id}/llm-responses")
async def get_report_llm_responses(report_id: int, request: Request):
    return await proxy_request(request, REPORT_SERVICE_URL, f"/api/reports/{report_id}/llm-responses")