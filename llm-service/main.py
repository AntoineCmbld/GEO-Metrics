import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import providers, health

# Add logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("LLM Service starting up...")
    logger.info(f"HuggingFace API Key configured: {bool(os.getenv('HUGGINGFACE_API_KEY'))}")
    yield
    # Shutdown
    logger.info("LLM Service shutting down...")

app = FastAPI(title="llm-service", version="0.1.0", lifespan=lifespan)

app.include_router(health.router, prefix="/api")
app.include_router(providers.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8081")),
        reload=True,
    )