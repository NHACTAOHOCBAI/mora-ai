# Monkey patch for Ragas compatibility with langchain_community 0.4.x
import sys
import types
try:
    from langchain_google_vertexai import ChatVertexAI
    # Create dummy modules
    chat_models_module = types.ModuleType("langchain_community.chat_models")
    vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_module.ChatVertexAI = ChatVertexAI
    
    # Register in sys.modules
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module
    sys.modules["langchain_community.chat_models"] = chat_models_module
except Exception:
    pass

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

# Startup events can be added here if needed in the future

