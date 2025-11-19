import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import health, reports, llm
from routes.proxy import close_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (if needed)
    yield
    # Shutdown logic
    await close_client()

app = FastAPI(title="api-gateway", version="0.1.0", lifespan=lifespan)

# Mount routers under /api
app.include_router(health.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(llm.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "3000")),
        reload=True,
    )