from typing import List
from pydantic import BaseModel, Field

class BenchmarkDatasetItem(BaseModel):
    question: str = Field(..., description="Câu hỏi dùng để đánh giá.")
    ground_truth: str = Field(..., description="Đáp án mẫu chính xác (ground truth).")
    generated_answer: str = Field(None, description="Câu trả lời thực tế đã sinh từ hệ thống.")
    retrieved_contexts: List[str] = Field(default_factory=list, description="Danh sách các ngữ cảnh thực tế đã truy xuất.")
    latency_ms: int = Field(0, description="Độ trễ xử lý thực tế.")


class BenchmarkEvaluateRequest(BaseModel):
    approach_name: str = Field(..., description="Tên của hướng tiếp cận (ví dụ: 'with_image_filter').")
    dataset: List[BenchmarkDatasetItem] = Field(..., description="Danh sách tập câu hỏi kiểm thử.")

class BenchmarkDetailResponse(BaseModel):
    question: str
    retrieved_contexts: str
    generated_answer: str
    groundTruth: str
    latencyMs: int
    faithfulness: float
    answerRelevance: float
    contextPrecision: float
    contextRecall: float

class BenchmarkEvaluateResponse(BaseModel):
    approachName: str
    faithfulness: float
    answerRelevance: float
    contextPrecision: float
    contextRecall: float
    avgLatencyMs: int
    details: List[BenchmarkDetailResponse]
