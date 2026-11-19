import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse, ChatSummarizeRequest, ChatSummarizeResponse
from app.services.chat_service import generate_chat_response, generate_chat_summary, generate_chat_response_stream
from app.services.openai_service import AI_ERROR, AI_CONFIGURATION_ERROR
from app.core.runtime_config import RuntimeConfigurationError
from app.core.usage_tracking import collect_usage
from app.core.stream_control import request_cancel

router = APIRouter()

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
