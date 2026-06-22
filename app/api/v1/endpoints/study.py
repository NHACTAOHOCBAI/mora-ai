from fastapi import APIRouter, HTTPException
from loguru import logger
from app.schemas.chat import StudyNotesRequest, StudyNotesResponse
from app.services.gemini_service import generate_study_notes_service

router = APIRouter()

@router.post("/notes", response_model=StudyNotesResponse)
async def generate_study_notes(request: StudyNotesRequest):
    try:
        return generate_study_notes_service(context=request.context)
    except Exception as e:
        logger.error(f"Error in generate_study_notes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")
