import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from clients.huggingface_client import HuggingFaceClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["providers"])

# Static list of available models
AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "zai-org/GLM-4.6"
]

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

class QueryRequest(BaseModel):
    prompt: str
    model: Optional[str] = None

class QueryResponse(BaseModel):
    model: str
    response: str

@router.get("/models")
async def list_models():
    """List available models for inference"""
    return {
        "count": len(AVAILABLE_MODELS),
        "models": AVAILABLE_MODELS
    }

@router.post("/query", response_model=QueryResponse)
async def query_provider(request: QueryRequest):
    """Query an LLM model with a prompt"""
    
    # Get API key
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="HuggingFace API key not configured")
    
    # Use default model if not specified
    model = request.model or DEFAULT_MODEL
    
    # Get available models from list_models endpoint
    models_response = await list_models()
    available_models = models_response["models"]
    
    # Check if model exists in available models
    if model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available. Available models: {', '.join(available_models)}"
        )
    
    try:
        client = HuggingFaceClient(api_key)
        result = client.query(request.prompt, model=model)
        return result
    except Exception as e:
        logger.error(f"Error querying HuggingFace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying HuggingFace: {str(e)}")