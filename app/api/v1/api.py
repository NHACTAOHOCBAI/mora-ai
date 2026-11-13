from fastapi import APIRouter
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.index import router as index_router
from app.api.v1.endpoints.parse import router as parse_router
from app.api.v1.endpoints.study import router as study_router

api_router = APIRouter()
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(index_router, tags=["index"])
api_router.include_router(parse_router, tags=["parse"])
api_router.include_router(study_router, tags=["study"])



