from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from clients.chatgpt_client import ChatGPTClient
from clients.groq_client import GroqClient

router = APIRouter(tags=["providers"])

class QueryRequest(BaseModel):
    provider: str
    prompt: str
    model: Optional[str] = None  # Optional model parameter

class QueryResponse(BaseModel):
    provider: str
    model: str
    response: str

class ProviderInfo(BaseModel):
    name: str
    available: bool
    models: List[str]

@router.get("/providers", response_model=List[ProviderInfo])
async def list_providers():
    providers = [
        {
            "name": "openai",
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        },
        {
            "name": "groq",
            "available": bool(os.getenv("GROQ_API_KEY")),
            "models": ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
        }
    ]
    return providers

@router.post("/query", response_model=QueryResponse)
async def query_provider(request: QueryRequest):
    provider = request.provider.lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key not configured")
        
        # Use provided model or default
        model = request.model or "gpt-3.5-turbo"
        client = ChatGPTClient(api_key)
        result = client.query(request.prompt, model=model)
        
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="Groq API key not configured")
        
        # Use provided model or default to llama-3.1-8b-instant
        model = request.model or "llama-3.1-8b-instant"
        client = GroqClient(api_key)
        result = client.query(request.prompt, model=model)
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    return result