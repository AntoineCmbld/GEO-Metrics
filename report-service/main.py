import os
from fastapi import FastAPI
from routes import reports

app = FastAPI(title="report-service", version="0.1.0")

app.include_router(reports.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )