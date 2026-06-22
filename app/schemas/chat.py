import base64
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class ChatMessageDto(BaseModel):
    sender: str = Field(..., description="Người gửi tin nhắn (user hoặc assistant).")
    text: str = Field(..., description="Nội dung chi tiết của tin nhắn.")

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
    context: str = Field(..., description="Ngữ cảnh văn bản từ tài liệu phục vụ cho mô hình AI.")
    question: str = Field(..., description="Câu hỏi của người dùng đặt cho tài liệu.")
    base64Image: Optional[str] = Field(default=None, description="Chuỗi ảnh Base64 đơn lẻ đi kèm (nếu có).")
    base64Images: List[str] = Field(default=[], description="Danh sách các chuỗi ảnh Base64 gửi kèm phục vụ đối chiếu.")
    history: List[ChatMessageDto] = Field(default=[], description="Lịch sử trò chuyện trước đó.")

    @field_validator("base64Image")
    @classmethod
    def validate_base64_image(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Dữ liệu ảnh Base64 không hợp lệ.")

    @field_validator("base64Images")
    @classmethod
    def validate_base64_images(cls, v: List[str]) -> List[str]:
        for img in v:
            try:
                base64.b64decode(img)
            except Exception:
                raise ValueError("Dữ liệu ảnh Base64 trong danh sách không hợp lệ.")
        return v

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
    context: str = Field(..., description="Siêu ngữ cảnh tổng hợp từ các tài liệu trong không gian học tập.")
    question: str = Field(..., description="Câu hỏi của người dùng đặt cho không gian học tập.")
    base64Images: List[str] = Field(default=[], description="Danh sách chuỗi ảnh Base64 của các trang tài liệu tương ứng.")
    history: List[ChatMessageDto] = Field(default=[], description="Lịch sử trò chuyện trước đó trong không gian học tập.")

    @field_validator("base64Images")
    @classmethod
    def validate_base64_images(cls, v: List[str]) -> List[str]:
        for img in v:
            try:
                base64.b64decode(img)
            except Exception:
                raise ValueError("Dữ liệu ảnh Base64 trong danh sách không hợp lệ.")
        return v

class Flashcard(BaseModel):
    question: str = Field(description="Câu hỏi ôn tập.")
    answer: str = Field(description="Đáp án của câu hỏi.")

class StudyNotesResponse(BaseModel):
    summary: str = Field(description="Tóm tắt nội dung tài liệu bằng Markdown.")
    flashcards: List[Flashcard] = Field(default=[], description="Danh sách các flashcards ôn tập.")

class StudyNotesRequest(BaseModel):
    context: str = Field(..., description="Ngữ cảnh văn bản từ tài liệu dùng để tạo tóm tắt và câu hỏi ôn tập.")
