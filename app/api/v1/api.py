from fastapi import APIRouter
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.study import router as study_router
from app.api.v1.endpoints.benchmark import router as benchmark_router

api_router = APIRouter()
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(study_router, prefix="/study", tags=["study"])
api_router.include_router(benchmark_router, prefix="/benchmark", tags=["benchmark"])

