from pydantic import BaseModel, Field

class DetectRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio")
    language: str = Field(..., description="ta | en | hi | ml | te")

class DetectResponse(BaseModel):
    classification: str = Field(..., example="AI_GENERATED")
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: str
    model_version: str
