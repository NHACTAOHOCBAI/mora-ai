import os
import time
import json
from typing import List, Dict, Any
from loguru import logger
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.schemas.benchmark import BenchmarkEvaluateRequest, BenchmarkEvaluateResponse, BenchmarkDetailResponse
from app.services.gemini_service import chat_with_document_service



def run_ragas_evaluation(request: BenchmarkEvaluateRequest) -> BenchmarkEvaluateResponse:
    logger.info(f"Bắt đầu chạy đánh giá Ragas cho hướng tiếp cận: {request.approach_name}")
    
    # Thiết lập API Key cho Langchain Google GenAI
    os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
    
    # Sử dụng model trọng tài từ cấu hình
    evaluator_llm = ChatGoogleGenerativeAI(
        model=settings.gemini_evaluator_model_name,
        temperature=0.0
    )
    
    # Sử dụng Google Generative AI Embeddings cho các metrics cần embedding
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_evaluator_embeddings_model_name
    )
    
    # Khởi tạo dữ liệu
    questions = []
    generated_answers = []
    retrieved_contexts_list = []
    ground_truths = []
    latencies = []
    
    details: List[BenchmarkDetailResponse] = []
    
    # 1. Thu thập dữ liệu từ request dataset đã qua xử lý thực tế ở Backend
    for item in request.dataset:
        logger.info(f"Đang chuẩn bị đánh giá Ragas cho câu hỏi: {item.question}")
        
        # Nếu chưa được backend xử lý đầy đủ, gán giá trị mặc định để tránh crash
        generated_answer = item.generated_answer if item.generated_answer else "Không có câu trả lời."
        retrieved_contexts = item.retrieved_contexts if item.retrieved_contexts else ["Không có ngữ cảnh."]
        latency_ms = item.latency_ms if item.latency_ms else 0

        logger.info(f" -> Ngữ cảnh đã truy xuất (retrieved_contexts): {retrieved_contexts}")
        logger.info(f" -> Câu trả lời đã sinh (generated_answer): {generated_answer}")
        logger.info(f" -> Thời gian phản hồi (latency): {latency_ms} ms")

        questions.append(item.question)
        generated_answers.append(generated_answer)
        retrieved_contexts_list.append(retrieved_contexts)
        ground_truths.append(item.ground_truth)
        latencies.append(latency_ms)
        
        # Tạo thông tin chi tiết tạm thời (chưa có điểm Ragas)
        details.append(BenchmarkDetailResponse(
            question=item.question,
            retrieved_contexts=json.dumps(retrieved_contexts, ensure_ascii=False),
            generated_answer=generated_answer,
            groundTruth=item.ground_truth,
            latencyMs=latency_ms,
            faithfulness=0.0,
            answerRelevance=0.0,
            contextPrecision=0.0,
            contextRecall=0.0
        ))
        
    # 2. Xây dựng Dataset cho Ragas
    data = {
        "question": questions,
        "contexts": retrieved_contexts_list,
        "answer": generated_answers,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 3. Gán LLM và Embeddings cho các chỉ số đánh giá của Ragas
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings
        
    # 4. Thực thi đánh giá bằng Ragas
    logger.info("Đang gọi Ragas để chấm điểm các chỉ số...")
    try:
        score_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        
        df_scores = score_result.to_pandas()
        
        import math
        def safe_float(val) -> float:
            try:
                f = float(val)
                return 0.0 if math.isnan(f) else f
            except (ValueError, TypeError):
                return 0.0

        # Cập nhật điểm số chi tiết cho từng câu hỏi
        for idx, row in df_scores.iterrows():
            details[idx].faithfulness = safe_float(row.get("faithfulness", 0.0))
            details[idx].answerRelevance = safe_float(row.get("answer_relevancy", 0.0))
            details[idx].contextPrecision = safe_float(row.get("context_precision", 0.0))
            details[idx].contextRecall = safe_float(row.get("context_recall", 0.0))
            
        overall_scores = score_result._repr_dict
        overall_faithfulness = safe_float(overall_scores.get("faithfulness", 0.0))
        overall_relevance = safe_float(overall_scores.get("answer_relevancy", 0.0))
        overall_precision = safe_float(overall_scores.get("context_precision", 0.0))
        overall_recall = safe_float(overall_scores.get("context_recall", 0.0))
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy Ragas evaluate: {e}", exc_info=True)
        # Trong trường hợp lỗi, trả về điểm mặc định
        overall_faithfulness = 0.0
        overall_relevance = 0.0
        overall_precision = 0.0
        overall_recall = 0.0
        
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0
    
    return BenchmarkEvaluateResponse(
        approachName=request.approach_name,
        faithfulness=overall_faithfulness,
        answerRelevance=overall_relevance,
        contextPrecision=overall_precision,
        contextRecall=overall_recall,
        avgLatencyMs=avg_latency,
        details=details
    )
