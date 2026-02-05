from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or human",
    version="1.0"
)

app.include_router(router)
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI Voice Detection API"
    }