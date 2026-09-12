from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger
from app.services.parser_service import parse_pdf_layout_and_diagrams

router = APIRouter()

@router.post("/parse")
async def parse_pdf_endpoint(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
    parser_model: Optional[str] = Form(None)
):
    logger.info(f"Received PDF parsing request: filename='{file.filename}', parser_model='{parser_model}'")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for layout parsing")
    
    try:
        content = await file.read()
        parsed_pages = parse_pdf_layout_and_diagrams(content, api_key=api_key, parser_model=parser_model)
        return {"status": "success", "pages": parsed_pages}
    except Exception as e:
        logger.error(f"Error occurred during PDF layout parsing: {e}")
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")

