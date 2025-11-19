from fastapi import APIRouter
from datetime import datetime, timezone
import os

router = APIRouter(tags=["health"])

@router.get("/health")
async def health():
    # Check if API keys are configured
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    
    providers_status = {
        "openai": "configured" if openai_configured else "not_configured",
        "groq": "configured" if groq_configured else "not_configured"
    }
    
    return {
        "status": "healthy",
        "service": "llm-service",
        "providers": providers_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }