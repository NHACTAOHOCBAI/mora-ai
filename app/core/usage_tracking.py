"""Request-local OpenAI usage collection for operational analytics.

Only counters and model metadata are collected; prompts and model output never
leave the request through this mechanism.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from app.core.config import settings

_events: ContextVar[list | None] = ContextVar("usage_events", default=None)


def _value(usage: Any, name: str) -> int:
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    return int(value or 0)


def record_usage(response: Any, model: str, operation: str, sink: list | None = None) -> dict | None:
    """Collect a provider usage record.

    ``sink`` is used by synchronous SSE generators, which may resume on a
    different worker thread and therefore cannot safely rely on ContextVar.
    """
    events = sink if sink is not None else _events.get()
    usage = getattr(response, "usage", None)
    if events is None or usage is None:
        return None
    input_tokens = _value(usage, "input_tokens")
    output_tokens = _value(usage, "output_tokens")
    total_tokens = _value(usage, "total_tokens") or input_tokens + output_tokens
    cached = _value(getattr(usage, "input_tokens_details", None), "cached_tokens")
    prices = settings.model_prices().get(model, settings.model_prices().get("default", {}))
    usd = (Decimal(input_tokens - cached) * Decimal(str(prices.get("input", 0)))
           + Decimal(cached) * Decimal(str(prices.get("cached_input", prices.get("input", 0))))
           + Decimal(output_tokens) * Decimal(str(prices.get("output", 0)))) / Decimal(1_000_000)
    cost_vnd = (usd * Decimal(str(settings.ai_usage_vnd_per_usd))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    item = {"operation": operation, "model": model, "inputTokens": input_tokens,
            "outputTokens": output_tokens, "cachedInputTokens": cached,
            "totalTokens": total_tokens, "estimatedCostVnd": float(cost_vnd),
            "pricingVersion": settings.ai_usage_pricing_version}
    events.append(item)
    return item


@contextmanager
def collect_usage():
    token = _events.set([])
    try:
        yield _events.get()
    finally:
        _events.reset(token)
