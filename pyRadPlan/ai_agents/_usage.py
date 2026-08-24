"""Logging of token usage and estimated cost for pydantic-ai agent runs."""

import logging
from typing import Any, Optional

from genai_prices import Usage as PriceUsage, calc_price

from ._settings import AiSettings

logger = logging.getLogger(__name__)

#: Usage summary of the most recent logged run, consumed via pop_last_run_usage.
_last_run_usage: list[str] = []


def summarize_run_usage(result: Any, model: str) -> Optional[str]:
    """Build a short usage/cost summary for a pydantic-ai run result.

    Parameters
    ----------
    result :
        The result returned by ``Agent.run_sync`` (exposes ``usage``).
    model : str
        The model identifier the agent ran with.

    Returns
    -------
    str or None
        A summary like ``"Tokens: 1200 in / 340 out | Est. cost: $0.0123"``,
        or ``None`` if the usage cannot be read from *result*.
    """
    try:
        usage = result.usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
    except Exception:
        logger.debug("Could not read usage from agent result", exc_info=True)
        return None

    cost_str = _estimate_cost(model, usage)
    return f"Tokens: {input_tokens} in / {output_tokens} out | Est. cost: {cost_str}"


def pop_last_run_usage() -> Optional[str]:
    """Return and clear the usage summary recorded by the most recent agent run.

    Lets callers that only receive an agent's domain output (e.g. the GUI's AI
    task dialog) display the usage of the run that produced it. Returns ``None``
    if no run has been logged since the last call, e.g. because
    ``AiSettings().display_usage`` is off or the usage could not be read.
    """
    return _last_run_usage.pop() if _last_run_usage else None


# Cost in USD because that's the default
def log_run_usage(result: Any, model: str, operation: Optional[str] = None) -> None:
    """Log token usage and estimated USD cost for a pydantic-ai run result.

    Does nothing if ``AiSettings().display_usage`` is ``False``. Otherwise the
    summary is also recorded for retrieval via :func:`pop_last_run_usage`.

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

    summary = summarize_run_usage(result, model)
    if summary is None:  # never let logging break an agent run
        return

    _last_run_usage[:] = [summary]

    label = f"AI agent ({operation})" if operation else "AI agent run"
    logger.info("%s | MODEL: %s | %s", label, model, summary)


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
