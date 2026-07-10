"""Tests for the beam-angle agent and its structure-set geometry context."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")

from pyRadPlan import load_tg119
from pyRadPlan.ai_agents import (
    beam_angles_system_prompt,
    cst_geometry_summary,
    generate_beam_angles,
)
from pyRadPlan.ai_agents._plan_agents import BEAM_ANGLES_GEOMETRY_PROMPT, BeamSetup
from pyRadPlan.plan import PhotonPlan


@pytest.fixture
def cst():
    return load_tg119()[1]


class _FakeAgent:
    """Stand-in for pydantic_ai.Agent capturing prompts and returning one beam."""

    captured: dict = {}

    def __init__(self, model, output_type=None, system_prompt=None):
        _FakeAgent.captured = {"model": model, "system_prompt": system_prompt}

    def run_sync(self, user_prompt):
        _FakeAgent.captured["user_prompt"] = user_prompt
        return SimpleNamespace(
            output=BeamSetup(gantry_angles=[0.0], couch_angles=[0.0]),
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr("pyRadPlan.ai_agents._plan_agents.Agent", _FakeAgent)
    monkeypatch.setattr("pyRadPlan.ai_agents._plan_agents.load_ai_env", lambda *a, **k: None)
    return _FakeAgent


def test_beam_angles_system_prompt_geometry_flag():
    plain = beam_angles_system_prompt("photons")
    assert BEAM_ANGLES_GEOMETRY_PROMPT not in plain
    with_geometry = beam_angles_system_prompt("photons", with_geometry=True)
    assert with_geometry.startswith(plain)
    assert BEAM_ANGLES_GEOMETRY_PROMPT in with_geometry


def test_cst_geometry_summary(cst):
    summary = cst_geometry_summary(cst)
    assert "LPS" in summary["coordinate_system"]
    assert len(summary["vois"]) == len(cst.vois)
    for entry in summary["vois"]:
        assert len(entry["center_of_mass_mm"]) == 3
        assert len(entry["principal_axes"]) == 3
        shape = entry["shape"]
        assert shape["volume_cc"] > 0
        assert len(shape["bounding_box_size_mm"]) == 3
        assert len(shape["equivalent_ellipsoid_diameters_mm"]) == 3
        assert shape["elongation"] >= 1.0
        assert shape["flatness"] >= 1.0


def test_generate_beam_angles_without_cst(fake_agent):
    pln = generate_beam_angles(PhotonPlan(), treatment_site="test site", model="test-model")

    assert fake_agent.captured["system_prompt"] == beam_angles_system_prompt("photons")
    assert "Structure set geometry" not in fake_agent.captured["user_prompt"]
    assert pln.prop_stf["gantry_angles"] == [0.0]


def test_generate_beam_angles_with_cst_geometry(cst, fake_agent):
    generate_beam_angles(PhotonPlan(), treatment_site="test site", model="test-model", cst=cst)

    assert fake_agent.captured["system_prompt"] == beam_angles_system_prompt(
        "photons", with_geometry=True
    )
    user_prompt = fake_agent.captured["user_prompt"]
    assert "Structure set geometry" in user_prompt
    for voi in cst.vois:
        assert voi.name in user_prompt
    assert "center_of_mass_mm" in user_prompt
