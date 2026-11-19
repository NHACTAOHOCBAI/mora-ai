"""Request-local diagnostics; ordinary chat keeps its existing response contract."""
from contextvars import ContextVar
import traceback
from loguru import logger

execution_trace: ContextVar[dict | None] = ContextVar("execution_trace", default=None)


def event(stage: str, **values):
    trace = execution_trace.get()
    if trace is not None:
        trace.setdefault(stage, {}).update(values)


def failure(stage: str, error: Exception):
    trace = execution_trace.get()
    if trace is not None:
        # Never persist provider error bodies: they may contain API keys or prompts.
        trace.setdefault("errors", []).append({"stage": stage, **error_summary(error)})


def error_summary(error: Exception):
    """Safe, actionable metadata for persisted traces; never store provider bodies."""
    result = {"type": type(error).__name__}
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(code, int):
        result["httpStatus"] = code
        result["reason"] = {
            400: "Provider rejected request parameters",
            401: "Provider API key authentication failed",
            403: "Provider denied access to the requested resource",
            404: "Requested provider model or resource was not found",
            429: "Provider quota or rate limit exceeded",
            500: "Provider internal error", 503: "Provider temporarily unavailable",
        }.get(code, "Provider request failed")
    return result


def log_benchmark_failure(stage: str, error: Exception):
    """Traceback without local variables, with configured credentials redacted."""
    from app.core.config import settings
    detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    for value in (settings.openai_api_key, settings.benchmark_judge_api_key, settings.qdrant_api_key):
        if value:
            detail = detail.replace(value, "[REDACTED]")
    # Pass provider text as an argument: JSON braces must never become log templates.
    logger.error("Benchmark {} failed:\n{}", stage, detail)
