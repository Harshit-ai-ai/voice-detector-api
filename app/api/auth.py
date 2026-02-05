from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from app.core.config import API_KEY, API_KEY_NAME
import os

API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    auto_error=False
)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )
    return api_key
