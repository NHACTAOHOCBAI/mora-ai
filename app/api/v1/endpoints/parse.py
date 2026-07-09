from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger
from app.services.parser_service import parse_pdf_layout_and_diagrams

router = APIRouter()

@router.post("/parse")
async def parse_pdf_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received PDF parsing request: filename='{file.filename}'")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for layout parsing")
    
    try:
        content = await file.read()
        parsed_pages = parse_pdf_layout_and_diagrams(content)
        return {"status": "success", "pages": parsed_pages}
    except Exception as e:
        logger.error(f"Error occurred during PDF layout parsing: {e}")
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
