import os
from fastapi import APIRouter, Request
from .proxy import proxy_request

router = APIRouter(prefix="/llm", tags=["llm"])

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8081")

@router.get("/models")
async def list_models(request: Request):
    """List available models"""
    target_url = f"{LLM_SERVICE_URL}/api/models"
    return await proxy_request(request, target_url)

@router.post("/query")
async def query_llm(request: Request):
    """Query an LLM provider"""
    target_url = f"{LLM_SERVICE_URL}/api/query"
    return await proxy_request(request, target_url)

@router.get("/health")
async def health_check(request: Request):
    """Check LLM service health"""
    target_url = f"{LLM_SERVICE_URL}/api/health"
    return await proxy_request(request, target_url)