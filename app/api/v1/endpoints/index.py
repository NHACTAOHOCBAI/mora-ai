from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger
from app.services.openai_service import get_client, embed

from app.services.chunking_service import extract_semantic_chunks_from_pages
from app.services.vector_service import vector_store

router = APIRouter()

class PageItem(BaseModel):
    pageNumber: int
    text: str

class IndexRequest(BaseModel):
    documentId: int
    spaceId: int
    documentName: str
    pages: List[PageItem]

@router.post("/index")
def index_endpoint(request: IndexRequest):
    logger.info(f"Bắt đầu lập chỉ mục Vector & BM25 cho Document '{request.documentName}' (ID: {request.documentId}, Space: {request.spaceId}, Pages: {len(request.pages)})")
    
    try:
        client = get_client()
    except Exception:
        raise HTTPException(status_code=503, detail="Dịch vụ AI chưa sẵn sàng.") from None

    # 1. Phân đoạn Semantic Chunks theo AST
    pages_dict = [{"pageNumber": p.pageNumber, "text": p.text} for p in request.pages]
    chunks = extract_semantic_chunks_from_pages(
        pages=pages_dict,
        space_id=request.spaceId,
        document_id=request.documentId,
        document_name=request.documentName
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="Tài liệu không có nội dung để lập chỉ mục.")

    embeddings = []
    try:
        for offset in range(0, len(chunks), 32):
            embeddings.extend(embed(client, [c.enrichedContent for c in chunks[offset:offset + 32]]))
    except Exception:
        raise HTTPException(status_code=503, detail="Không thể tạo chỉ mục tài liệu lúc này.") from None

    # 3. Nạp vào Qdrant Container và BM25 Index
    try:
        vector_store.upsert_chunks(
            space_id=request.spaceId,
            document_id=request.documentId,
            chunks=chunks,
            embeddings=embeddings
        )
        logger.info(f"Lập chỉ mục thành công cho Document ID {request.documentId} vào Qdrant & BM25!")
    except Exception as e:
        logger.error(f"Lỗi khi nạp chunks vào Qdrant: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lưu chỉ mục tài liệu. Vui lòng thử lại sau.")

    return {
        "status": "success", 
        "message": f"Document {request.documentId} indexed successfully",
        "chunksCount": len(chunks)
    }
