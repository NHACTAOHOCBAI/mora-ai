from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse, ChatSummarizeRequest, ChatSummarizeResponse
from app.services.chat_service import generate_chat_response, generate_chat_summary

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return generate_chat_response(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/summarize", response_model=ChatSummarizeResponse)
def chat_summarize_endpoint(request: ChatSummarizeRequest):
    try:
        return generate_chat_summary(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
