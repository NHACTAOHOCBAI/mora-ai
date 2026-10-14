from fastapi import APIRouter, HTTPException
from loguru import logger
from app.schemas.chat import (
    ChatRequest, ChatResponse, 
    ChatSummarizeRequest, ChatSummarizeResponse,
    ChatValidateKeyRequest, ChatValidateKeyResponse
)
from app.services.chat_service import generate_chat_response, generate_chat_summary, validate_gemini_key

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return generate_chat_response(request)
    except ValueError as ve:
        logger.warning("Chat request rejected: {}", ve)
        raise HTTPException(status_code=400, detail="Yêu cầu trò chuyện không hợp lệ hoặc chưa có API Key.")
    except Exception as e:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=500, detail="Dịch vụ AI đang gặp sự cố. Vui lòng thử lại sau.")

@router.post("/chat/summarize", response_model=ChatSummarizeResponse)
def chat_summarize_endpoint(request: ChatSummarizeRequest):
    try:
        return generate_chat_summary(request)
    except Exception as e:
        logger.exception("Chat summary generation failed")
        raise HTTPException(status_code=500, detail="Không thể tóm tắt cuộc trò chuyện lúc này.")

@router.post("/chat/validate-key", response_model=ChatValidateKeyResponse)
def validate_key_endpoint(request: ChatValidateKeyRequest):
    valid, message = validate_gemini_key(request.api_key, request.model_name)
    return ChatValidateKeyResponse(valid=valid, message=message)
