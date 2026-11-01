from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse, ChatSummarizeRequest, ChatSummarizeResponse
from app.services.chat_service import generate_chat_response, generate_chat_summary
from app.services.openai_service import AI_ERROR

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return generate_chat_response(request)
    except Exception:
        raise HTTPException(status_code=503, detail=AI_ERROR) from None

@router.post("/chat/summarize", response_model=ChatSummarizeResponse)
def chat_summarize_endpoint(request: ChatSummarizeRequest):
    try:
        return generate_chat_summary(request)
    except Exception:
        raise HTTPException(status_code=503, detail=AI_ERROR) from None
