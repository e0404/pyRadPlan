"""Tests for the objective-suggestion agent and its QI-based adaptation."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")

from pyRadPlan import load_tg119
from pyRadPlan.analysis import DX, Mean, QICollection, StructureQIs
from pyRadPlan.ai_agents import (
    OBJECTIVES_ADAPT_PROMPT,
    OBJECTIVES_SYSTEM_PROMPT,
    cst_context_summary,
    generate_voi_objectives,
    objectives_system_prompt,
)
from pyRadPlan.plan import PhotonPlan


@pytest.fixture
def cst():
    return load_tg119()[1]


@pytest.fixture
def pln():
    return PhotonPlan()


def _make_qis(cst) -> QICollection:
    """QICollection with a finite mean and a NaN D50 for every VOI."""
    structures = {
        voi.name: StructureQIs(
            name=voi.name,
            metrics={
                "mean": Mean(value=1.5),
                "D50": DX(value=float("nan"), ref_vol=50.0),
            },
        )
        for voi in cst.vois
    }
    return QICollection(structures=structures)


class _FakeAgent:
    """Stand-in for pydantic_ai.Agent capturing prompts and returning no objectives."""

    captured: dict = {}

    def __init__(self, model, output_type=None, system_prompt=None):
        _FakeAgent.captured = {
            "model": model,
            "output_type": output_type,
            "system_prompt": system_prompt,
        }

    def run_sync(self, user_prompt):
        _FakeAgent.captured["user_prompt"] = user_prompt
        output = _FakeAgent.captured["output_type"](objectives=[])
        return SimpleNamespace(
            output=output, usage=SimpleNamespace(input_tokens=0, output_tokens=0)
        )


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr("pyRadPlan.ai_agents._cst_agents.Agent", _FakeAgent)
    monkeypatch.setattr("pyRadPlan.ai_agents._cst_agents.load_ai_env", lambda *a, **k: None)
    return _FakeAgent


def test_objectives_system_prompt_adapt_flag():
    assert objectives_system_prompt() == OBJECTIVES_SYSTEM_PROMPT
    adapted = objectives_system_prompt(adapt=True)
    assert adapted.startswith(OBJECTIVES_SYSTEM_PROMPT)
    assert OBJECTIVES_ADAPT_PROMPT in adapted


def test_cst_context_summary_without_qis(pln, cst):
    summary = cst_context_summary(pln, cst)
    assert "quality_indicators" not in summary
    assert "current_objectives" not in summary
    assert len(summary["vois"]) == len(cst.vois)


def test_cst_context_summary_with_qis(pln, cst):
    qis = _make_qis(cst)
    summary = cst_context_summary(pln, cst, qis=qis)

    assert set(summary["quality_indicators"]) == {voi.name for voi in cst.vois}
    metrics = summary["quality_indicators"][cst.vois[0].name]
    assert metrics["mean"] == "1.5 Gy"
    # NaN metrics are dropped from the context sent to the model
    assert "D50" not in metrics

    objectives = summary["current_objectives"]
    assert set(objectives) == {voi.name for voi in cst.vois}
    voi_with_objectives = next(voi for voi in cst.vois if voi.objectives)
    assert (
        objectives[voi_with_objectives.name][0]["name"] == voi_with_objectives.objectives[0].name
    )


def test_generate_voi_objectives_without_qis_prompts(pln, cst, fake_agent):
    generate_voi_objectives(pln, cst, treatment_site="test site", model="test-model")

    assert fake_agent.captured["system_prompt"] == OBJECTIVES_SYSTEM_PROMPT
    assert "Quality indicators" not in fake_agent.captured["user_prompt"]
    assert "suggest one or more objectives" in fake_agent.captured["user_prompt"]


def test_generate_voi_objectives_with_qis_adapts_prompts(pln, cst, fake_agent):
    qis = _make_qis(cst)
    new_cst = generate_voi_objectives(
        pln, cst, treatment_site="test site", model="test-model", qis=qis
    )

    system_prompt = fake_agent.captured["system_prompt"]
    assert system_prompt == objectives_system_prompt(adapt=True)

    user_prompt = fake_agent.captured["user_prompt"]
    # the previous objectives are captured before they are cleared for the run
    voi_with_objectives = next(voi for voi in cst.vois if voi.objectives)
    assert voi_with_objectives.objectives[0].name in user_prompt
    assert "1.5 Gy" in user_prompt
    assert "Adapt the previous objectives" in user_prompt

    # fake output holds no suggestions -> objectives cleared, input untouched
    assert all(not voi.objectives for voi in new_cst.vois)
    assert any(voi.objectives for voi in cst.vois)


def test_generate_voi_objectives_rejects_mismatched_qis(pln, cst, fake_agent):
    qis = QICollection(
        structures={"NotAVoi": StructureQIs(name="NotAVoi", metrics={"mean": Mean(value=1.0)})}
    )
    with pytest.raises(ValueError, match="match a VOI"):
        generate_voi_objectives(pln, cst, treatment_site="test site", model="test-model", qis=qis)
