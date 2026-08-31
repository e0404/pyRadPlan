"""Tests for usage logging and recall of agent runs."""

from types import SimpleNamespace

from pyRadPlan._settings import get_settings
from pyRadPlan.ai.agents import pop_last_run_usage
from pyRadPlan.ai.agents._usage import log_run_usage, summarize_run_usage


def _fake_result(input_tokens=1200, output_tokens=340):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    )


def test_summarize_run_usage_formats_tokens_and_cost():
    summary = summarize_run_usage(_fake_result(), "no-such-provider:no-such-model")
    assert "1200 in / 340 out" in summary
    assert "n/a" in summary  # unknown model cannot be priced


def test_summarize_run_usage_handles_missing_usage():
    assert summarize_run_usage(object(), "some-model") is None


def test_log_run_usage_records_last_usage(monkeypatch):
    monkeypatch.setattr(get_settings().ai, "agents_display_usage", True)
    pop_last_run_usage()  # clear leftovers from other tests

    log_run_usage(_fake_result(), "no-such-provider:no-such-model", operation="test")

    summary = pop_last_run_usage()
    assert summary is not None
    assert "1200 in / 340 out" in summary
    assert pop_last_run_usage() is None  # consumed


def test_log_run_usage_respects_display_usage_off(monkeypatch):
    monkeypatch.setattr(get_settings().ai, "agents_display_usage", False)
    pop_last_run_usage()

    log_run_usage(_fake_result(), "no-such-provider:no-such-model")

    assert pop_last_run_usage() is None
