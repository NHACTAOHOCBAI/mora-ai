"""Server-owned OpenAI access. Never accept credentials from requests."""
import hashlib
import time

from openai import OpenAI, OpenAIError
from loguru import logger
from app.core.config import settings
from app.core.runtime_config import refresh_runtime_config, RuntimeConfigurationError
from app.core.usage_tracking import record_usage
from app.core.stream_control import attach_response, is_cancel_requested

AI_ERROR = "Dịch vụ AI đang gặp sự cố. Vui lòng thử lại sau."
AI_CONFIGURATION_ERROR = "Dịch vụ AI chưa được cấu hình. Vui lòng liên hệ quản trị viên."

def get_client():
    refresh_runtime_config()
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        logger.error("Admin AI configuration did not include a provider key")
        raise RuntimeConfigurationError("Provider key missing from Admin configuration")
    key = settings.openai_api_key.strip()
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    logger.info("OpenAI credential loaded (length={}, sha256={})", len(key), fingerprint)
    return OpenAI(api_key=key, timeout=settings.openai_timeout,
                  max_retries=settings.openai_max_retries)

def generate(client, model, prompt, instructions, schema=None, operation="chat", usage_sink=None):
    kwargs = dict(model=model, input=prompt, instructions=instructions, store=False)
    try:
        if schema:
            response = client.responses.parse(**kwargs, text_format=schema)
            if response.status != "completed" or response.output_parsed is None:
                logger.error("OpenAI structured response chưa hoàn tất hoặc bị từ chối.")
                raise RuntimeError(AI_ERROR)
            record_usage(response, model, operation, sink=usage_sink)
            return response.output_parsed.model_dump_json()
        response = client.responses.create(**kwargs)
        if response.status != "completed" or not response.output_text.strip():
            logger.error("OpenAI response chưa hoàn tất hoặc không có nội dung.")
            raise RuntimeError(AI_ERROR)
        record_usage(response, model, operation, sink=usage_sink)
        return response.output_text
    except OpenAIError as error:
        logger.error("OpenAI request failed: type={}, HTTP={}", type(error).__name__,
                     getattr(error, "status_code", None))
        safe_error = RuntimeError(AI_ERROR)
        safe_error.status_code = getattr(error, "status_code", None)
        raise safe_error from None

def stream_text(client, model, prompt, instructions, operation="chat_generation", usage_sink=None, stream_control=None):
    """Yield text deltas from Responses API without buffering the full answer."""
    response_id = None
    terminal_response = None
    try:
        with client.responses.stream(model=model, input=prompt, instructions=instructions,
                                     background=True, store=False) as stream:
            for event in stream:
                if event.type == "response.created":
                    response_id = event.response.id
                    attach_response(stream_control, response_id)
                if event.type == "response.output_text.delta" and not is_cancel_requested(stream_control):
                    yield event.delta
            terminal_response = stream.get_final_response()
    except OpenAIError as error:
        logger.error("OpenAI streaming request failed: type={}, HTTP={}", type(error).__name__,
                     getattr(error, "status_code", None))
        if not response_id:
            raise RuntimeError(AI_ERROR) from None
    finally:
        # A background response remains retrievable after a client disconnect,
        # cancellation, or interrupted stream.  Its terminal object is OpenAI's
        # source of truth for charged token usage.
        if response_id:
            for _ in range(120):
                try:
                    terminal_response = client.responses.retrieve(response_id)
                    if terminal_response.status not in {"queued", "in_progress"}:
                        break
                except OpenAIError as error:
                    logger.warning("Could not retrieve OpenAI response {}: {}", response_id, type(error).__name__)
                time.sleep(0.25)
            if terminal_response is not None:
                record_usage(terminal_response, model, operation, sink=usage_sink)

def embed(client, texts, operation="embedding", usage_sink=None):
    response = client.embeddings.create(model=settings.openai_embedding_model,
                                        input=texts, dimensions=1536)
    items = sorted(response.data, key=lambda item: item.index)
    if len(items) != len(texts) or any(len(item.embedding) != 1536 for item in items):
        raise RuntimeError(AI_ERROR)
    record_usage(response, settings.openai_embedding_model, operation, sink=usage_sink)
    return [item.embedding for item in items]
