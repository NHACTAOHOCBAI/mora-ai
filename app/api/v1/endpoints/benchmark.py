from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, ConfigDict
from app.core.config import settings
from app.core.execution_trace import log_benchmark_failure
from app.core.usage_tracking import collect_usage
from app.services.benchmark_service import ExecuteSample, ScoreSample, CorpusRequest, configuration, execute, score, prepare_corpus


router = APIRouter(prefix="/internal/benchmarks")

class BenchmarkIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documentId: int
    spaceId: int
    documentName: str
    pages: list[dict]


@router.post("/parse")
async def parse_endpoint(file: UploadFile = File(...)):
    if not settings.openai_api_key:
        raise HTTPException(503, "Configure OPENAI_API_KEY on AI Service")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "PDF required")
    from app.services.parser_service import parse_pdf_layout_and_diagrams
    from starlette.concurrency import run_in_threadpool
    try:
        with collect_usage() as usage:
            pages = await run_in_threadpool(parse_pdf_layout_and_diagrams, await file.read())
            return {"status": "success", "pages": pages, "usage": usage}
    except Exception as error:
        log_benchmark_failure("parse", error)
        raise HTTPException(502, "Benchmark PDF parsing failed")


@router.post("/index")
def index_endpoint(request: BenchmarkIndexRequest):
    if not settings.openai_api_key:
        raise HTTPException(503, "Configure OPENAI_API_KEY on AI Service")
    from app.api.v1.endpoints.index import IndexRequest, index_endpoint as index_document
    return index_document(IndexRequest(**request.model_dump()))


@router.post("/prepare")
def prepare_endpoint(request: CorpusRequest):
    try:
        return prepare_corpus(request)
    except Exception as error:
        log_benchmark_failure("prepare", error)
        raise HTTPException(409, "Benchmark corpus unavailable")


@router.get("/config")
def config_endpoint():
    try:
        return configuration()
    except Exception as error:
        log_benchmark_failure("config", error)
        raise HTTPException(503, "Benchmark configuration or dependencies unavailable")


@router.post("/execute")
def execute_endpoint(request: ExecuteSample):
    try:
        with collect_usage() as usage:
            result = execute(request)
            return {**result, "usage": usage}
    except ValueError:
        raise HTTPException(409, "Benchmark configuration mismatch")
    except Exception as error:
        log_benchmark_failure("execute", error)
        raise HTTPException(502, "Benchmark pipeline failed")


@router.post("/score")
async def score_endpoint(request: ScoreSample):
    try:
        with collect_usage() as usage:
            result = await score(request)
            return {**result, "usage": usage}
    except ValueError:
        raise HTTPException(409, "Benchmark configuration mismatch")
    except Exception as error:
        log_benchmark_failure("score", error)
        raise HTTPException(502, "Benchmark judge failed")
