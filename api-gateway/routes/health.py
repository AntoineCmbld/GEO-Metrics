from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/health")

@router.get("/")
async def health():
    return {"status": "ok", "service": "api-gateway", "time": datetime.utcnow().isoformat() + "Z"}