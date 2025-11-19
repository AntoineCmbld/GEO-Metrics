from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
from clients.chatgpt_client import ChatGPTClient
from clients.groq_client import GroqClient

router = APIRouter(tags=["providers"])

class QueryRequest(BaseModel):
    provider: str
    prompt: str

class QueryResponse(BaseModel):
    provider: str
    response: str
    score: float

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
            "models": ["gpt-3.5-turbo", "gpt-4"]
        },
        {
            "name": "groq",
            "available": bool(os.getenv("GROQ_API_KEY")),
            "models": ["llama2-70b-4096"]
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
        client = ChatGPTClient(api_key)
        result = client.query(request.prompt)
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="Groq API key not configured")
        client = GroqClient(api_key)
        result = client.query(request.prompt)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    return result

@router.get("/providers/{name}/models", response_model=List[str])
async def list_provider_models(name: str):
    models_map = {
        "openai": ["gpt-3.5-turbo", "gpt-4"],
        "groq": ["llama2-70b-4096"]
    }
    
    if name.lower() not in models_map:
        raise HTTPException(status_code=404, detail=f"Provider not found: {name}")
    
    return models_map[name.lower()]