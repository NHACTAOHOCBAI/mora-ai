from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger
from google import genai

from app.core.config import settings
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
    apiKey: Optional[str] = None

@router.post("/index")
def index_endpoint(request: IndexRequest):
    logger.info(f"Bắt đầu lập chỉ mục Vector & BM25 cho Document '{request.documentName}' (ID: {request.documentId}, Space: {request.spaceId}, Pages: {len(request.pages)})")
    
    api_key = request.apiKey or settings.gemini_api_key
    if not api_key:
        logger.error("Không tìm thấy Gemini API Key để tạo Embeddings.")
        raise HTTPException(status_code=400, detail="Gemini API Key is required for generating embeddings.")

    # 1. Phân đoạn Semantic Chunks theo AST
    pages_dict = [{"pageNumber": p.pageNumber, "text": p.text} for p in request.pages]
    chunks = extract_semantic_chunks_from_pages(
        pages=pages_dict,
        space_id=request.spaceId,
        document_id=request.documentId,
        document_name=request.documentName
    )

    if not chunks:
        logger.warn(f"Không có chunk nào được trích xuất từ tài liệu ID: {request.documentId}")
        return {"status": "success", "message": "No text content found to index", "chunksCount": 0}

    # 2. Tạo Vector Embeddings qua Google GenAI SDK với cơ chế Retry & Rate-limit
    logger.info(f"Đang tạo Embeddings cho {len(chunks)} chunks sử dụng model '{settings.gemini_embedding_model}'...")
    client = genai.Client(api_key=api_key)
    embeddings = []

    import time
    try:
        for idx, chunk in enumerate(chunks):
            for attempt in range(5):
                try:
                    resp = client.models.embed_content(
                        model=settings.gemini_embedding_model,
                        contents=chunk.enrichedContent
                    )
                    val = resp.embeddings[0].values if resp.embeddings else []
                    embeddings.append(val)
                    time.sleep(0.05)
                    break
                except Exception as api_err:
                    if "429" in str(api_err) or "RESOURCE_EXHAUSTED" in str(api_err):
                        logger.warning(f"Rate limit (429) khi tạo embedding chunk {idx+1}/{len(chunks)}. Đang chờ 5s (thử lại lần {attempt+1})...")
                        time.sleep(5)
                    else:
                        raise api_err

        logger.info(f"Tạo thành công {len(embeddings)} vector embeddings!")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tạo Gemini Embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Failed to upsert to Qdrant: {str(e)}")

    return {
        "status": "success", 
        "message": f"Document {request.documentId} indexed successfully",
        "chunksCount": len(chunks)
    }
