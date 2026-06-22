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

# Mẫu ngữ cảnh và hình ảnh giả định cho việc kiểm thử RAG
MOCK_CONTEXT = (
    "Mora là một nền tảng hỗ trợ học tập thông minh tích hợp trí tuệ nhân tạo (AI).\n"
    "Hệ thống cho phép người dùng tải lên tài liệu học tập (PDF, Hình ảnh), tự động tóm tắt nội dung "
    "và tạo ra các câu hỏi ôn tập (Flashcards). Mora sử dụng PostgreSQL với extension pgvector làm "
    "Cơ sở dữ liệu Vector để lưu trữ và tìm kiếm ngữ cảnh tài liệu tương đồng.\n"
    "Hệ thống AI của Mora được xây dựng trên nền tảng Gemini API của Google. Đối với các tác vụ thông thường, "
    "Mora sử dụng mô hình gemini-1.5-flash để cân bằng giữa chi phí và tốc độ phản hồi. Đối với tác vụ chấm điểm "
    "đánh giá chất lượng (Benchmark), hệ thống sử dụng mô hình gemini-2.5-flash làm trọng tài."
)

# Mock base64 image (a tiny transparent pixel)
MOCK_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

def execute_rag_pipeline(question: str, approach_name: str) -> Dict[str, Any]:
    """
    Chạy pipeline RAG thực tế dựa theo hướng tiếp cận (approach_name)
    để đo lường và so sánh hiệu năng của các giải pháp khác nhau.
    """
    start_time = time.time()
    
    # Giả lập các hướng tiếp cận khác nhau để xử lý ảnh hoặc prompt
    base64_images = [MOCK_IMAGE_BASE64]
    
    if "With_Image_Filtering" in approach_name:
        # Hướng tiếp cận: Lọc bỏ hình ảnh để giảm chi phí/thời gian nếu câu hỏi không cần ảnh
        logger.info(f"[{approach_name}] Áp dụng lọc ảnh: loại bỏ ảnh ra khỏi request")
        base64_images = []
    elif "No_Images" in approach_name:
        logger.info(f"[{approach_name}] Chạy cấu hình không dùng ảnh")
        base64_images = []
    else:
        logger.info(f"[{approach_name}] Giữ nguyên hình ảnh mặc định")

    # Gọi hàm xử lý chat thực tế từ gemini_service
    # Ở đây chúng ta sử dụng ngữ cảnh học tập mẫu
    chat_response = chat_with_document_service(
        context=MOCK_CONTEXT,
        question=question,
        base64_images=base64_images,
        history=[]
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Giả lập RAG Contexts trả về cho Ragas đánh giá
    retrieved_contexts = [MOCK_CONTEXT]
    
    return {
        "generated_answer": chat_response.answer,
        "retrieved_contexts": retrieved_contexts,
        "latency_ms": latency_ms
    }

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
    
    # 1. Chạy từng test case trong bộ dataset
    for item in request.dataset:
        logger.info(f"Đang xử lý câu hỏi: {item.question}")
        rag_result = execute_rag_pipeline(item.question, request.approach_name)
        
        questions.append(item.question)
        generated_answers.append(rag_result["generated_answer"])
        retrieved_contexts_list.append(rag_result["retrieved_contexts"])
        ground_truths.append(item.ground_truth)
        latencies.append(rag_result["latency_ms"])
        
        # Tạo thông tin chi tiết tạm thời (chưa có điểm Ragas)
        details.append(BenchmarkDetailResponse(
            question=item.question,
            retrieved_contexts=json.dumps(rag_result["retrieved_contexts"], ensure_ascii=False),
            generated_answer=rag_result["generated_answer"],
            latencyMs=rag_result["latency_ms"],
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
