from typing import List, Optional
from pydantic import BaseModel, Field

class ChatMessageDto(BaseModel):
    sender: str
    text: str

class DocumentCitation(BaseModel):
    quote: str = Field(description="Đoạn văn bản trích dẫn chính xác từ tài liệu.")
    pageNumber: int = Field(description="Số trang chứa trích dẫn.")

class DocumentChatResponse(BaseModel):
    answerFound: bool = Field(description="True nếu tìm thấy câu trả lời trong tài liệu, false nếu không tìm thấy.")
    answer: str = Field(description="Nội dung câu trả lời hoặc câu từ chối trả lời.")
    citations: List[DocumentCitation] = Field(default=[], description="Danh sách các trích dẫn nguồn từ tài liệu.")
    condensedQuestion: Optional[str] = Field(default=None, description="Câu hỏi đã rút gọn độc lập.")
    promptSent: Optional[str] = Field(default=None, description="Prompt thực tế được gửi cho Gemini.")

class DocumentChatRequest(BaseModel):
    context: str
    question: str
    base64Image: Optional[str] = None
    base64Images: List[str] = []
    history: List[ChatMessageDto] = []

class SpaceCitation(BaseModel):
    quote: str = Field(description="Đoạn văn bản trích dẫn chính xác từ tài liệu.")
    pageNumber: int = Field(description="Số trang chứa trích dẫn.")
    documentId: int = Field(description="ID của tài liệu chứa trích dẫn.")

class SpaceChatResponse(BaseModel):
    answerFound: bool = Field(description="True nếu tìm thấy câu trả lời trong tài liệu, false nếu không tìm thấy.")
    answer: str = Field(description="Nội dung câu trả lời hoặc câu từ chối trả lời.")
    citations: List[SpaceCitation] = Field(default=[], description="Danh sách các trích dẫn nguồn từ tài liệu.")
    condensedQuestion: Optional[str] = Field(default=None, description="Câu hỏi đã rút gọn độc lập.")
    promptSent: Optional[str] = Field(default=None, description="Prompt thực tế được gửi cho Gemini.")

class SpaceChatRequest(BaseModel):
    context: str
    question: str
    base64Images: List[str] = []
    history: List[ChatMessageDto] = []

class Flashcard(BaseModel):
    question: str = Field(description="Câu hỏi ôn tập.")
    answer: str = Field(description="Đáp án của câu hỏi.")

class StudyNotesResponse(BaseModel):
    summary: str = Field(description="Tóm tắt nội dung tài liệu bằng Markdown.")
    flashcards: List[Flashcard] = Field(default=[], description="Danh sách các flashcards ôn tập.")

class StudyNotesRequest(BaseModel):
    context: str
