import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, OpenAIError
from app.schemas.chat import ChatRequest, ChatResponse, ChatSummarizeRequest, ChatSummarizeResponse
from app.services.chat_service import generate_chat_response, generate_chat_summary, generate_chat_response_stream
from app.services.openai_service import AI_ERROR, AI_CONFIGURATION_ERROR, get_client
from app.core.config import settings
from app.core.runtime_config import RuntimeConfigurationError
from app.core.usage_tracking import collect_usage
from app.core.stream_control import request_cancel

router = APIRouter()

def _validation_message(error: OpenAIError) -> str:
    status = getattr(error, "status_code", None)
    if status == 401:
        return "API key không hợp lệ, đã hết hạn hoặc đã bị thu hồi."
    if status == 403:
        return "API key không có quyền dùng model này."
    if status == 404:
        return "Model không tồn tại hoặc không khả dụng với API key này."
    if status == 429:
        return "Tài khoản đã chạm hạn mức hoặc quota của OpenAI."
    return "Không thể xác minh với OpenAI lúc này."

@router.post("/config/validate")
def validate_runtime_configuration():
    """Verify stored credentials and model access without exposing provider secrets."""
    checks = []
    try:
        client = get_client()
    except RuntimeConfigurationError:
        return {"valid": False, "checks": [{"name": "Kết nối Mora", "valid": False, "message": AI_CONFIGURATION_ERROR}]}

    configs = [
        ("Chat", client, [settings.openai_model, settings.openai_parser_model, settings.openai_embedding_model]),
        ("Benchmark", OpenAI(api_key=settings.benchmark_judge_api_key.strip(), timeout=settings.openai_timeout,
                               max_retries=0), [settings.benchmark_judge_model, settings.benchmark_judge_embedding_model]),
    ]
    for name, provider, models in configs:
        try:
            for model in dict.fromkeys(model for model in models if model):
                provider.models.retrieve(model)
            checks.append({"name": name, "valid": True, "message": "API key và các model đã được OpenAI chấp nhận."})
        except OpenAIError as error:
            checks.append({"name": name, "valid": False, "message": _validation_message(error)})
        except Exception:
            checks.append({"name": name, "valid": False, "message": "Không thể xác minh với OpenAI lúc này."})
    return {"valid": all(check["valid"] for check in checks), "checks": checks}

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        with collect_usage() as usage:
            response = generate_chat_response(request)
            response.usage = usage
            return response
    except RuntimeConfigurationError:
        raise HTTPException(status_code=503, detail=AI_CONFIGURATION_ERROR) from None
    except Exception:
        raise HTTPException(status_code=503, detail=AI_ERROR) from None

@router.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    def sse():
        usage = []
        try:
            for event_name, payload in generate_chat_response_stream(request, usage_sink=usage):
                yield f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except RuntimeConfigurationError:
            yield f"event: error\ndata: {json.dumps({'message': AI_CONFIGURATION_ERROR}, ensure_ascii=False)}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': AI_ERROR}, ensure_ascii=False)}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@router.delete("/chat/stream/{stream_id}")
def cancel_chat_stream(stream_id: str):
    request_cancel(stream_id)
    return {"status": "cancellation_requested"}

@router.post("/chat/summarize", response_model=ChatSummarizeResponse)
def chat_summarize_endpoint(request: ChatSummarizeRequest):
    try:
        with collect_usage() as usage:
            response = generate_chat_summary(request)
            response.usage = usage
            return response
    except RuntimeConfigurationError:
        raise HTTPException(status_code=503, detail=AI_CONFIGURATION_ERROR) from None
    except Exception:
        raise HTTPException(status_code=503, detail=AI_ERROR) from None
