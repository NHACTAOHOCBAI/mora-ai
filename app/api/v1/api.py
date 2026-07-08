from fastapi import APIRouter
from app.api.v1.endpoints.benchmark import router as benchmark_router

api_router = APIRouter()
api_router.include_router(benchmark_router, prefix="/benchmark", tags=["benchmark"])


