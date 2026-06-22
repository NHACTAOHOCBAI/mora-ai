from fastapi import APIRouter, HTTPException
from loguru import logger
from app.schemas.benchmark import BenchmarkEvaluateRequest, BenchmarkEvaluateResponse
from app.services.benchmark_service import run_ragas_evaluation

router = APIRouter()

@router.post("/evaluate", response_model=BenchmarkEvaluateResponse)
async def evaluate_benchmark(request: BenchmarkEvaluateRequest):
    logger.info(f"Received evaluation request for approach: {request.approach_name}")
    try:
        if not request.dataset:
            raise HTTPException(status_code=400, detail="Danh sách câu hỏi kiểm thử không được để trống.")
        
        response = run_ragas_evaluation(request)
        return response
    except Exception as e:
        logger.error(f"Error in evaluate_benchmark API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi đánh giá Ragas: {str(e)}")
