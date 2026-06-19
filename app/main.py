from fastapi import FastAPI
from app.api.v1.api import api_router

app = FastAPI(
    title="Mora AI Service",
    description="Microservice to handle Gemini API operations for Mora social learning network.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Đăng ký router tổng hợp với prefix /api để giữ nguyên tương thích URL với backend
app.include_router(api_router, prefix="/api")
