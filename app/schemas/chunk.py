from pydantic import BaseModel
from typing import Optional

class SemanticChunk(BaseModel):
    chunkId: str
    documentId: int
    spaceId: int
    documentName: Optional[str] = None
    pageNumber: int
    sectionPath: str          # Ví dụ: "Chương 1: Tổng quan > 1.1 Khái niệm HĐH"
    chunkType: str            # "TEXT", "TABLE", "DIAGRAM"
    content: str              # Nội dung văn bản thực tế của chunk
    enrichedContent: str      # Nội dung đính kèm Breadcrumb Header phục vụ tạo Vector Embedding
    tokenCount: int
