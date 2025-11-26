from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
from datetime import datetime, timezone

router = APIRouter(tags=["health"])

@router.get("/health")
async def health(
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    db_status = "unknown"
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Log the user ID
    print(f"Health check by user_id: {x_user_id}")

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "service": "report-service",
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": x_user_id
    }