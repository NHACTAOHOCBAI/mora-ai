from pydantic import BaseModel, Field
from typing import List, Optional

class ChatContextItem(BaseModel):
    pageNumber: int
    text: str
    documentName: Optional[str] = None
    documentId: Optional[int] = None

class ChatHistoryItem(BaseModel):
    sender: str  # "user" or "assistant"
    text: str

class ChatRequest(BaseModel):
    question: str
    context: List[ChatContextItem]
    history: List[ChatHistoryItem]

class Citation(BaseModel):
    pageNumber: int
    quote: str
    documentId: Optional[int] = None
    documentName: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    condensedQuestion: Optional[str] = None
    promptSent: Optional[str] = None
