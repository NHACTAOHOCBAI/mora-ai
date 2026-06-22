from typing import List
from pydantic import BaseModel, Field

class BenchmarkDatasetItem(BaseModel):
    question: str = Field(..., description="Câu hỏi dùng để đánh giá.")
    ground_truth: str = Field(..., description="Đáp án mẫu chính xác (ground truth).")

class BenchmarkEvaluateRequest(BaseModel):
    approach_name: str = Field(..., description="Tên của hướng tiếp cận (ví dụ: 'with_image_filter').")
    dataset: List[BenchmarkDatasetItem] = Field(..., description="Danh sách tập câu hỏi kiểm thử.")

class BenchmarkDetailResponse(BaseModel):
    question: str
    retrieved_contexts: str
    generated_answer: str
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
