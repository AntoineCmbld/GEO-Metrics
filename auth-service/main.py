import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import login, register

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Auth Service starting up...")
    yield
    logger.info("Auth Service shutting down...")

app = FastAPI(
    title="auth-service",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(login.router, prefix="/api")
app.include_router(register.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )