import os
from fastapi import APIRouter, Request, Depends
from fastapi.security import HTTPBearer
from .proxy import proxy_request
from .auth_utils import get_user_id

security = HTTPBearer()
router = APIRouter(prefix="/llm", tags=["llm"], dependencies=[Depends(security)])

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8081")

@router.get("/models")
async def list_models(request: Request, user_id: int = Depends(get_user_id)):
    """List available models"""
    target_url = f"{LLM_SERVICE_URL}/api/models"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)

@router.get("/health")
async def health_check(request: Request, user_id: int = Depends(get_user_id)):
    """Check LLM service health"""
    target_url = f"{LLM_SERVICE_URL}/api/health"
    headers = {"X-User-ID": str(user_id)}
    return await proxy_request(request, target_url, extra_headers=headers)