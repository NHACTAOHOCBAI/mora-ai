"""Benchmark-only adapters. Ground truth is accepted only by score(), never execute()."""
import asyncio
import importlib.metadata
import math
import hashlib
import json
import subprocess
import platform
import time
from pathlib import Path
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from loguru import logger
from app.core.config import settings
from app.core.runtime_config import refresh_runtime_config
from app.core.execution_trace import execution_trace, error_summary, log_benchmark_failure, failure, event
from app.schemas.chat import ChatRequest, ChatContextItem
from app.services.chat_service import generate_chat_response


class ExecuteSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    spaceId: int
    documentIds: list[int] = Field(min_length=1)
    context: list[ChatContextItem] = Field(default_factory=list)
    config: dict
    corpusFingerprint: str


class CorpusRequest(BaseModel):
    spaceId: int
    documentIds: list[int] = Field(min_length=1)
    context: list[ChatContextItem] = Field(default_factory=list)


def prepare_corpus(request: CorpusRequest):
    """Prepare an OpenAI corpus; saved pages allow repairing missing documents before a run."""
    from app.services.vector_service import vector_store
    from app.schemas.chunk import SemanticChunk
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    if not vector_store.client:
        raise ValueError("Qdrant unavailable")
    chunks = []
    offset = None
    while True:
        points, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=Filter(must=[FieldCondition(key="space_id", match=MatchValue(value=request.spaceId)),
                                      FieldCondition(key="document_id", match=MatchAny(any=request.documentIds))]),
            limit=256, offset=offset, with_payload=True, with_vectors=False)
        for point in points:
            p = point.payload
            chunks.append(SemanticChunk(
                chunkId=p["chunk_id"], documentId=p["document_id"], spaceId=request.spaceId,
                documentName=p.get("document_name", ""), pageNumber=p["page_number"],
                sectionPath=p.get("section_path", ""), chunkType=p.get("chunk_type", "TEXT"),
                content=p.get("content", ""), enrichedContent=p.get("enriched_content", ""), tokenCount=0))
        if offset is None:
            break
    missing = set(request.documentIds) - {c.documentId for c in chunks}
    if missing:
        # Only initial prepare receives saved pages. Execute/resume pass no pages,
        # so a frozen run never silently changes its indexed corpus.
        pages_by_document = {document_id: [] for document_id in missing}
        for page in request.context:
            if page.documentId in pages_by_document:
                pages_by_document[page.documentId].append(page)
        if any(not pages for pages in pages_by_document.values()):
            raise ValueError("Missing OpenAI chunks for document IDs: " + ", ".join(map(str, sorted(missing))))
        from app.api.v1.endpoints.index import IndexRequest, PageItem, index_endpoint
        for document_id, pages in sorted(pages_by_document.items()):
            pages.sort(key=lambda page: page.pageNumber)
            index_endpoint(IndexRequest(
                documentId=document_id, spaceId=request.spaceId,
                documentName=pages[0].documentName or str(document_id),
                pages=[PageItem(pageNumber=page.pageNumber, text=page.text) for page in pages]
            ))
        # Re-read and fingerprint the actual stored chunks, never the input pages.
        return prepare_corpus(CorpusRequest(spaceId=request.spaceId, documentIds=request.documentIds))
    chunks.sort(key=lambda c: (c.documentId, c.chunkId))
    manifest = [c.model_dump() for c in chunks]
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if getattr(vector_store, "_benchmark_fingerprint", None) != fingerprint:
        # Dedicated benchmark space: reconstruct precisely this corpus in stable order.
        from rank_bm25 import BM25Okapi
        vector_store.bm25_store[request.spaceId] = {
            "chunks": chunks, "bm25": BM25Okapi([c.enrichedContent.lower().split() for c in chunks])}
        vector_store._benchmark_fingerprint = fingerprint
    return {"fingerprint": fingerprint, "chunkCount": len(chunks)}


class ScoreSample(BaseModel):
    question: str
    referenceAnswer: str
    trace: dict
    config: dict


def configuration():
    refresh_runtime_config()
    if not settings.openai_api_key or not settings.benchmark_judge_api_key:
        raise ValueError("Admin AI configuration is incomplete")
    def version(package):
        return importlib.metadata.version(package)
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "chatModel": settings.openai_model,
        "parserModel": settings.openai_parser_model,
        "routerModel": settings.openai_model,
        "evaluatorModel": settings.openai_model,
        "judgeModel": settings.benchmark_judge_model,
        "judgeEmbeddingModel": settings.benchmark_judge_embedding_model,
        "judgeAdapter": "openai-responses-v1",
        "metricDetailsVersion": 1,
        "judgeTimeout": settings.benchmark_judge_timeout,
        "judgeMaxRetries": settings.benchmark_judge_max_retries,
        "embeddingModel": settings.openai_embedding_model,
        "topK": settings.retrieval_top_k, "denseLimit": settings.retrieval_dense_limit,
        "sparseLimit": settings.retrieval_sparse_limit,
        "rrfK": 60, "candidateLimit": 20, "rankerModel": "ms-marco-TinyBERT-L-2-v2",
        "maxSynthesisAttempts": 2, "retrievalEnabled": settings.retrieval_enabled,
        "providerMaxRetries": settings.openai_max_retries, "providerTimeout": settings.openai_timeout,
        "versions": {p: version(p) for p in ("ragas", "openai", "qdrant-client", "flashrank", "instructor", "langchain-community")},
        "pythonVersion": platform.python_version(),
        "aiCommit": commit,
    }


def verify_config(config):
    if config != configuration():
        raise ValueError("Server configuration changed; create a new benchmark run")


def execute(request: ExecuteSample):
    verify_config(request.config)
    corpus = prepare_corpus(CorpusRequest(spaceId=request.spaceId, documentIds=request.documentIds))
    if corpus["fingerprint"] != request.corpusFingerprint:
        raise ValueError("Indexed corpus changed")
    trace = {"errors": []}
    token = execution_trace.set(trace)
    started = time.perf_counter()
    try:
        response = generate_chat_response(ChatRequest(
            question=request.question, space_id=request.spaceId, document_ids=request.documentIds,
            context=request.context, history=[], chat_summary="",
        ))
        trace["response"] = response.model_dump()
        return trace
    except Exception as error:
        failure("pipeline", error)
        event("pipeline", status="error", timings={"totalMs": (time.perf_counter() - started) * 1000})
        return trace
    finally:
        execution_trace.reset(token)


def metric_value(value=None, status="ok", reason=None):
    return {"value": value, "status": status, "reason": reason}


async def score(request: ScoreSample):
    verify_config(request.config)
    from app.services.benchmark_judge import ResponsesJudge
    from app.services.benchmark_details import RecordingJudge, RecordingEmbeddings, details_for
    from ragas.metrics.collections import ContextPrecision, Faithfulness, AnswerRelevancy
    refresh_runtime_config()
    client = AsyncOpenAI(api_key=settings.benchmark_judge_api_key,
                         timeout=settings.benchmark_judge_timeout,
                         max_retries=settings.benchmark_judge_max_retries)
    try:
        llm = RecordingJudge(ResponsesJudge(client, request.config["judgeModel"]))
        embeddings = RecordingEmbeddings(client=client, model=request.config["judgeEmbeddingModel"])
        context_items = request.trace.get("pipeline", {}).get("contexts", [])
        contexts = [c["text"] for c in context_items]
        answer = request.trace["response"]["answer"]
        output = {}
        calls = {
            "contextPrecision": lambda: ContextPrecision(llm=llm).ascore(
                user_input=request.question, reference=request.referenceAnswer, retrieved_contexts=contexts),
            "faithfulness": lambda: Faithfulness(llm=llm).ascore(
                user_input=request.question, response=answer, retrieved_contexts=contexts),
            "answerRelevance": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
                user_input=request.question, response=answer),
        }
        for name, call in calls.items():
            llm.outputs.clear()
            embeddings.original = None
            embeddings.generated = []
            started = time.perf_counter()
            if not contexts and name != "answerRelevance":
                output[name] = metric_value(0.0) if name == "contextPrecision" else metric_value(
                    status="not_applicable", reason="No context supplied to RAG synthesis")
                output[name]["details"] = details_for(name, [], embeddings, context_items, True)
                output[name]["details"]["durationMs"] = 0
                continue
            try:
                started = time.perf_counter()
                logger.info("Benchmark judge metric={} started; contexts={}, timeout=180s", name, len(contexts))
                result = await asyncio.wait_for(call(), timeout=180)
                value = float(result.value)
                if not math.isfinite(value):
                    raise ValueError("Non-finite metric")
                output[name] = metric_value(value)
                logger.info("Benchmark judge metric={} completed in {:.1f}s", name, time.perf_counter() - started)
            except Exception as error:
                summary = error_summary(error)
                log_benchmark_failure("judge." + name, error)
                output[name] = metric_value(status="error", reason=summary.get("reason", summary["type"]))
                if "httpStatus" in summary:
                    output[name]["httpStatus"] = summary["httpStatus"]
            output[name]["details"] = details_for(name, llm.outputs, embeddings, context_items,
                                                  output[name]["status"] == "ok")
            output[name]["details"]["durationMs"] = (time.perf_counter() - started) * 1000
        return output
    finally:
        await client.close()
