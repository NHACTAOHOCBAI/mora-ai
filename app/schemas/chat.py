from pydantic import BaseModel, Field
from typing import List, Optional

class ChatContextItem(BaseModel):
    pageNumber: int
    text: str
    documentName: Optional[str] = None
    documentId: Optional[int] = None
    sectionPath: Optional[str] = None

class ChatHistoryItem(BaseModel):
    sender: str  # "user" or "assistant"
    text: str

class ChatRequest(BaseModel):
    question: str
    context: Optional[List[ChatContextItem]] = []
    history: List[ChatHistoryItem]
    chat_summary: Optional[str] = None
    api_key: Optional[str] = None
    chat_model: Optional[str] = None
    router_model: Optional[str] = None
    evaluator_model: Optional[str] = None
    space_id: Optional[int] = None
    spaceId: Optional[int] = None
    document_id: Optional[int] = None
    documentId: Optional[int] = None

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

class ChatSummarizeRequest(BaseModel):
    history: List[ChatHistoryItem]
    previous_summary: Optional[str] = None
    api_key: Optional[str] = None
    summarizer_model: Optional[str] = None

class ChatSummarizeResponse(BaseModel):
    summary: str

class ChatValidateKeyRequest(BaseModel):
    api_key: str
    model_name: Optional[str] = "gemini-2.5-flash"

class ChatValidateKeyResponse(BaseModel):
    valid: bool
    message: str
