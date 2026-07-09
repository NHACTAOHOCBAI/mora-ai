from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger

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
    logger.info(f"Indexing document: name='{request.documentName}', id={request.documentId}, pages={len(request.pages)}")
    # Mocking chunking and embedding generation for the pipeline
    return {"status": "success", "message": f"Document {request.documentId} indexed successfully"}
