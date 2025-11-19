import os
from fastapi import APIRouter, Request
from .proxy import proxy_request

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8081")

router = APIRouter(prefix="/llm")

@router.get("/health")
async def llm_health(request: Request):
    return await proxy_request(request, LLM_SERVICE_URL, "/api/health")

@router.get("/providers")
async def list_providers(request: Request):
    return await proxy_request(request, LLM_SERVICE_URL, "/api/providers")

@router.post("/query")
async def query_provider(request: Request):
    return await proxy_request(request, LLM_SERVICE_URL, "/api/query")