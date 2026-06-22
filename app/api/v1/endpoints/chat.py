from fastapi import APIRouter, HTTPException
from loguru import logger
from app.schemas.chat import (
    DocumentChatRequest,
    DocumentChatResponse,
    SpaceChatRequest,
    SpaceChatResponse
)
from app.services.gemini_service import chat_with_document_service, chat_with_space_service

router = APIRouter()

@router.post("/document", response_model=DocumentChatResponse)
async def chat_with_document(request: DocumentChatRequest):
    try:
        images = list(request.base64Images) if request.base64Images else []
        if request.base64Image:
            images.append(request.base64Image)
        return chat_with_document_service(
            context=request.context,
            question=request.question,
            base64_images=images,
            history=request.history
        )
    except ValueError as e:
        logger.warning(f"Validation error in chat_with_document: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_with_document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

@router.post("/space", response_model=SpaceChatResponse)
async def chat_with_space(request: SpaceChatRequest):
    try:
        return chat_with_space_service(
            context=request.context,
            question=request.question,
            base64_images=request.base64Images,
            history=request.history
        )
    except ValueError as e:
        logger.warning(f"Validation error in chat_with_space: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_with_space: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")
