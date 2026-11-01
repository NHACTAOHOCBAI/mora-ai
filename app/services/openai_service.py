"""Server-owned OpenAI access. Never accept credentials from requests."""
from openai import OpenAI, OpenAIError
from loguru import logger
from app.core.config import settings

AI_ERROR = "Dịch vụ AI đang gặp sự cố. Vui lòng thử lại sau."

def get_client():
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        logger.error("OPENAI_API_KEY chưa được nạp. Kiểm tra mora-ai/.env và khởi động lại AI Service.")
        raise RuntimeError(AI_ERROR)
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout,
                  max_retries=settings.openai_max_retries)

def generate(client, model, prompt, instructions, schema=None):
    kwargs = dict(model=model, input=prompt, instructions=instructions, store=False)
    try:
        if schema:
            response = client.responses.parse(**kwargs, text_format=schema)
            if response.status != "completed" or response.output_parsed is None:
                logger.error("OpenAI structured response chưa hoàn tất hoặc bị từ chối.")
                raise RuntimeError(AI_ERROR)
            return response.output_parsed.model_dump_json()
        response = client.responses.create(**kwargs)
        if response.status != "completed" or not response.output_text.strip():
            logger.error("OpenAI response chưa hoàn tất hoặc không có nội dung.")
            raise RuntimeError(AI_ERROR)
        return response.output_text
    except OpenAIError as error:
        logger.error("OpenAI request failed: type={}, HTTP={}", type(error).__name__,
                     getattr(error, "status_code", None))
        raise RuntimeError(AI_ERROR) from None

def embed(client, texts):
    response = client.embeddings.create(model=settings.openai_embedding_model,
                                        input=texts, dimensions=1536)
    items = sorted(response.data, key=lambda item: item.index)
    if len(items) != len(texts) or any(len(item.embedding) != 1536 for item in items):
        raise RuntimeError(AI_ERROR)
    return [item.embedding for item in items]
