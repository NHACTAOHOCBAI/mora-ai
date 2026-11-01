from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger
from app.services.parser_service import parse_pdf_layout_and_diagrams

router = APIRouter()

@router.post("/parse")
def parse_pdf_endpoint(
    file: UploadFile = File(...)
):
    logger.info(f"Received PDF parsing request: filename='{file.filename}'")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp PDF.")
    
    try:
        content = file.file.read()
        parsed_pages = parse_pdf_layout_and_diagrams(content)
        return {"status": "success", "pages": parsed_pages}
    except Exception:
        logger.error("PDF parsing failed")
        raise HTTPException(status_code=503, detail="Không thể xử lý tài liệu PDF. Vui lòng thử lại sau.")
