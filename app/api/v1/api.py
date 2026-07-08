from fastapi import APIRouter
from app.api.v1.endpoints.benchmark import router as benchmark_router
from app.api.v1.endpoints.chat import router as chat_router

api_router = APIRouter()
api_router.include_router(benchmark_router, prefix="/benchmark", tags=["benchmark"])
api_router.include_router(chat_router, tags=["chat"])



