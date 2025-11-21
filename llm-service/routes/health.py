from fastapi import APIRouter
from datetime import datetime, timezone
import os

router = APIRouter(tags=["health"])

@router.get("/health")
async def health():
    huggingface_configured = bool(os.getenv("HUGGINGFACE_API_KEY"))
    
    providers_status = {
        "huggingface": "configured" if huggingface_configured else "not_configured"
    }
    
    return {
        "status": "healthy",
        "service": "llm-service",
        "providers": providers_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }