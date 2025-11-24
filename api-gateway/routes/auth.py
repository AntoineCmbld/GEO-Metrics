import os
from fastapi import APIRouter, Request
from .proxy import proxy_request

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8080")

@router.post("/register")
async def register(request: Request):
    target_url = f"{AUTH_SERVICE_URL}/api/auth/register"
    return await proxy_request(request, target_url)

@router.post("/login")
async def login(request: Request):
    target_url = f"{AUTH_SERVICE_URL}/api/auth/login"
    return await proxy_request(request, target_url)