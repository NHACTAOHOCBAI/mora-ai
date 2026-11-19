"""Load the customer-facing AI configuration from the backend control plane."""
import json
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from loguru import logger
from app.core.config import settings

class RuntimeConfigurationError(RuntimeError):
    """The mandatory Admin-owned AI configuration cannot be loaded."""

    pass

def refresh_runtime_config() -> None:
    """Apply the latest admin configuration for this new AI request.

    There is deliberately no .env or cached-value fallback: Admin is the sole
    source of provider credentials, models and pricing.
    """
    internal_key = (settings.ai_service_internal_key or "").strip()
    if not internal_key:
        raise RuntimeConfigurationError("AI service internal authentication is not configured")
    request = Request(
        f"{settings.backend_url.rstrip('/')}/api/internal/ai-settings/runtime",
        headers={"X-AI-Service-Key": internal_key}, method="GET")
    try:
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        settings.openai_api_key = payload["apiKey"]
        settings.openai_model = payload["chatModel"]
        settings.openai_parser_model = payload["parserModel"]
        settings.openai_embedding_model = payload["embeddingModel"]
        settings.benchmark_judge_api_key = payload["benchmarkApiKey"]
        settings.benchmark_judge_model = payload["benchmarkJudgeModel"]
        settings.benchmark_judge_embedding_model = payload["benchmarkJudgeEmbeddingModel"]
        settings.ai_usage_vnd_per_usd = float(payload["vndPerUsd"])
        settings.ai_usage_pricing_version = payload["pricingVersion"]
        settings.runtime_model_prices = {
            name: {"input": float(price["inputPerMillionUsd"]),
                   "cached_input": float(price["cachedInputPerMillionUsd"]),
                   "output": float(price["outputPerMillionUsd"])}
            for name, price in payload["modelPrices"].items()
        }
    except (HTTPError, URLError, KeyError, ValueError, json.JSONDecodeError) as error:
        logger.warning("AI runtime config unavailable; AI requests are blocked ({})", type(error).__name__)
        raise RuntimeConfigurationError("Admin AI configuration is unavailable") from None
