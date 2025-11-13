from fastapi import FastAPI
from app.api.routes import analyze

app = FastAPI(title="GEO Metrics Multi-AI")

# Include routes
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
