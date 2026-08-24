"""Tests for the generic AI task dialog."""

import time

import pytest

pytest.importorskip("PySide6")

from pyRadPlan.gui.widgets.ai import AiTask, AiTaskDialog


def _pump(qapp, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_dialog_runs_task_and_applies_result(qapp):
    applied = []
    task = AiTask(
        title="Test task",
        system_prompt="system",
        context_text="context",
        run=lambda model, site, ctx: {"model": model, "site": site, "ctx": ctx},
        apply=applied.append,
        summarize=lambda r: f"got {r['model']}",
    )
    dialog = AiTaskDialog(task, ["model-a"])
    dialog._txt_site.setText("prostate")
    dialog._txt_context.setPlainText("more info")

    dialog._on_run()
    _pump(qapp, lambda: bool(applied))

    assert applied[0] == {"model": "model-a", "site": "prostate", "ctx": "more info"}
    assert "got model-a" in dialog._lbl_status.text()


def test_dialog_shows_usage_after_run(qapp, monkeypatch):
    """Usage recorded during the run (as the agent functions do) ends up in the status."""
    from types import SimpleNamespace

    from pyRadPlan.ai_agents._usage import log_run_usage

    monkeypatch.setenv("PYRADPLAN_AI_DISPLAY_USAGE", "true")
    applied = []

    def _run(model, site, ctx):
        fake_result = SimpleNamespace(usage=SimpleNamespace(input_tokens=1200, output_tokens=340))
        log_run_usage(fake_result, model, operation="test")
        return "result"

    task = AiTask(
        title="Test task",
        system_prompt="system",
        context_text="context",
        run=_run,
        apply=applied.append,
        summarize=lambda r: "Applied.",
    )
    dialog = AiTaskDialog(task, ["model-a"])
    dialog._on_run()
    _pump(qapp, lambda: bool(applied))

    status = dialog._lbl_status.text()
    assert "Applied." in status
    assert "1200 in / 340 out" in status


def test_dialog_requires_a_model(qapp):
    task = AiTask(
        title="Test",
        system_prompt="system",
        context_text="context",
        run=lambda *a: pytest.fail("run must not be called without a model"),
        apply=lambda r: None,
    )
    dialog = AiTaskDialog(task, [])
    dialog._cmb_model.setEditText("")
    dialog._on_run()
    assert "model" in dialog._lbl_status.text().lower()


def test_dialog_surfaces_errors(qapp):
    def _boom(model, site, ctx):
        raise RuntimeError("backend down")

    task = AiTask(
        title="Test",
        system_prompt="system",
        context_text="context",
        run=_boom,
        apply=lambda r: None,
    )
    dialog = AiTaskDialog(task, ["model-a"])
    dialog._on_run()
    _pump(qapp, lambda: "backend down" in dialog._lbl_status.text())

    assert "backend down" in dialog._lbl_status.text()
