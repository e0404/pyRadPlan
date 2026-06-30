"""Logging of token usage and estimated cost for pydantic-ai agent runs."""

import logging
from typing import Any, Optional

from genai_prices import Usage as PriceUsage, calc_price

from ._settings import AiSettings

logger = logging.getLogger(__name__)


# Cost in USD because that's the default
def log_run_usage(result: Any, model: str, operation: Optional[str] = None) -> None:
    """Log token usage and estimated USD cost for a pydantic-ai run result.

    Does nothing if ``AiSettings().display_usage`` is ``False``.

    Parameters
    ----------
    result :
        The result returned by ``Agent.run_sync`` (exposes ``usage``).
    model : str
        The model identifier the agent ran with.
    operation : str, optional
        Name of the calling operation, included in the log line for context.
    """
    if not AiSettings().display_usage:
        return

    try:
        usage = result.usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
    except Exception:  # never let logging break an agent run
        logger.debug("Could not read usage from agent result", exc_info=True)
        return

    cost_str = _estimate_cost(model, usage)

    label = f"AI agent ({operation})" if operation else "AI agent run"
    logger.info(
        "%s | MODEL: %s | TOKENS: %d/%d | EST. COST: %s",
        label,
        model,
        input_tokens,
        output_tokens,
        cost_str,
    )


def _estimate_cost(model: str, usage: Any) -> str:
    """Estimate the USD cost of a run, returning ``"n/a"`` if it cannot be priced."""
    # pydantic-ai model strings may be "provider:model" (e.g. "openai:gpt-4o").
    if ":" in model:
        provider_id, model_ref = model.split(":", 1)
    else:
        provider_id, model_ref = None, model

    try:
        price = calc_price(
            PriceUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_tokens", None) or 0,
                cache_write_tokens=getattr(usage, "cache_write_tokens", None) or 0,
            ),
            model_ref=model_ref,
            provider_id=provider_id,
        )
        return f"${price.total_price:.4f}"
    except Exception:
        logger.debug("Could not estimate price for model %r", model, exc_info=True)
        return "n/a"
