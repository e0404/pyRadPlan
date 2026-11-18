"""AIBeamletEngine.is_available and engine registration.

The engine is framework-agnostic: what a model needs is declared by the model
itself (``metadata.requirements``), so these tests fake a missing package rather
than a missing torch, and run on installs without any ML framework.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import pyRadPlan

from pyRadPlan._settings import get_settings
from pyRadPlan.dose.engines import AIBeamletEngine, get_available_engines, get_engine
from pyRadPlan.dose.engines import _base_ai_beamlet
from pyRadPlan.dose.engines._factory import DOSE_ENGINES
from pyRadPlan.plan import validate_pln

DUMMY_MODEL = "DKFZ-RadOpt/dummy_dosecalc_model"  # trained for protons


@pytest.fixture(autouse=True)
def _local_models_offline(monkeypatch):
    models_dir = Path(__file__).resolve().parents[1] / "data" / "ai_models"
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", models_dir)
    monkeypatch.setattr(get_settings().ai, "modelhub_offline", True)


@pytest.fixture
def unmet_requirements(monkeypatch):
    """Pretend the packages the model declares are not installed."""
    monkeypatch.setattr(_base_ai_beamlet, "model_facts", lambda model_dir: ({}, None, ["torch"]))


@pytest.fixture
def met_requirements(monkeypatch):
    """Pretend the packages the model declares are installed."""
    model_facts = _base_ai_beamlet.model_facts
    monkeypatch.setattr(
        _base_ai_beamlet, "model_facts", lambda model_dir: (*model_facts(model_dir)[:2], [])
    )


def _pln(radiation_mode="protons", model=DUMMY_MODEL):
    prop = {"engine": "AIBeamlet"}
    if model is not None:
        prop["model"] = model
    return validate_pln(
        {"radiation_mode": radiation_mode, "machine": "Generic", "prop_dose_calc": prop}
    )


def test_registered():
    assert DOSE_ENGINES["AIBeamlet"] is AIBeamletEngine


def test_model_framework_missing(unmet_requirements):
    """A model whose declared packages are missing is not offered (nor is the engine)."""
    available, msg = AIBeamletEngine.is_available(_pln(), None)
    assert available is False
    assert "torch" in msg
    assert "AIBeamlet" not in get_available_engines("protons")
    assert "HongPB" in get_available_engines("protons")


def test_model_available(met_requirements):
    assert AIBeamletEngine.is_available(_pln(), None) == (True, "")
    assert "AIBeamlet" in get_available_engines(_pln())


def test_model_missing_offline():
    available, msg = AIBeamletEngine.is_available(_pln(model="other_org/no_such_model"), None)
    assert available is False
    assert "no_such_model" in msg
    assert "AIBeamlet" not in get_available_engines(_pln(model="other_org/no_such_model"))


def test_model_missing_locally_but_on_hub(monkeypatch):
    monkeypatch.setattr(
        _base_ai_beamlet, "list_hub_models", lambda org, offline=False: [f"{org}/remote_model"]
    )
    assert AIBeamletEngine.is_available(_pln(model="other_org/remote_model"), None) == (True, "")


def test_model_missing_and_no_network(monkeypatch):
    monkeypatch.setattr(_base_ai_beamlet, "list_hub_models", lambda org, offline=False: [])
    available, _ = AIBeamletEngine.is_available(_pln(model="other_org/remote_model"), None)
    assert available is False


def test_bare_model_name_is_rejected():
    """Models are addressed as '<org>/<model>'; a bare name is not resolved."""
    available, msg = AIBeamletEngine.is_available(_pln(model="dummy_dosecalc_model"), None)
    assert available is False
    assert "<org>/<model>" in msg


def test_model_facts(tmp_path):
    config = {"metadata": {"training": {"modality": "VHEE"}, "requirements": ["no_such_package"]}}
    (tmp_path / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert _base_ai_beamlet.model_facts(tmp_path) == (config, "VHEE", ["no_such_package"])

    # the older modality key still works, and a model declaring nothing needs nothing
    older = {"metadata": {"dose_calc": {"radiation_mode": "protons"}}}
    (tmp_path / "model_config.json").write_text(json.dumps(older), encoding="utf-8")
    assert _base_ai_beamlet.model_facts(tmp_path) == (older, "protons", [])

    assert _base_ai_beamlet.model_facts(tmp_path / "missing") == ({}, None, [])


def test_config_choices_lists_local_and_hub_models(monkeypatch):
    monkeypatch.setattr(_base_ai_beamlet, "list_local_models", lambda: ["org_a/local_model"])
    monkeypatch.setattr(
        _base_ai_beamlet, "list_hub_models", lambda org, offline=False: ["org_b/hub_model"]
    )
    monkeypatch.setattr(_base_ai_beamlet, "local_model_dir", lambda name: Path(name))
    monkeypatch.setattr(
        _base_ai_beamlet,
        "model_facts",
        lambda model_dir: ({"metadata": {"dose_calc": {}}}, "protons", []),
    )
    monkeypatch.setattr(_base_ai_beamlet, "task_from_dir", lambda model_dir: None)

    assert AIBeamletEngine.config_choices("protons") == {
        "model": ["org_a/local_model", "org_b/hub_model"]
    }


def test_config_choices_only_offers_models_the_engine_could_run(monkeypatch):
    """Outcome models, other modalities and missing frameworks are left out."""
    facts = {
        "org/dosecalc-protons": ({"metadata": {"dose_calc": {}}}, "protons", []),
        "org/dosecalc-vhee": ({"metadata": {"dose_calc": {}}}, "VHEE", []),
        "org/dosecalc-no-framework": ({"metadata": {"dose_calc": {}}}, "protons", ["torch"]),
        "org/outcome-model": ({"metadata": {}}, None, []),
        "org/unreadable": ({}, None, []),
    }
    monkeypatch.setattr(_base_ai_beamlet, "list_local_models", lambda: list(facts))
    monkeypatch.setattr(_base_ai_beamlet, "list_hub_models", lambda org, offline=False: [])
    monkeypatch.setattr(_base_ai_beamlet, "local_model_dir", lambda name: Path(name))
    monkeypatch.setattr(
        _base_ai_beamlet, "model_facts", lambda model_dir: facts[Path(model_dir).as_posix()]
    )
    monkeypatch.setattr(_base_ai_beamlet, "task_from_dir", lambda model_dir: None)

    assert AIBeamletEngine.config_choices("protons") == {"model": ["org/dosecalc-protons"]}
    assert AIBeamletEngine.config_choices("VHEE") == {"model": ["org/dosecalc-vhee"]}
    # without a radiation mode only the task and the framework filter
    assert AIBeamletEngine.config_choices()["model"] == [
        "org/dosecalc-protons",
        "org/dosecalc-vhee",
    ]


def test_hub_models_not_listed_when_offline():
    assert _base_ai_beamlet.list_hub_models("DKFZ-RadOpt", True) == ()


def test_modality_mismatch():
    available, msg = AIBeamletEngine.is_available(_pln(radiation_mode="VHEE"), None)
    assert available is False
    assert "'protons'" in msg and "'VHEE'" in msg


def test_no_model_named_uses_local_models(met_requirements):
    assert AIBeamletEngine.is_available("protons", None) == (True, "")
    assert AIBeamletEngine.is_available(_pln(model=None), None) == (True, "")
    available, msg = AIBeamletEngine.is_available("VHEE", None)
    assert available is False
    assert "VHEE" in msg


def test_no_models_on_disk_but_hub_has_some(monkeypatch, tmp_path):
    """A fresh installation must still see the engine, so a model can be picked at all."""
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(get_settings().ai, "modelhub_offline", False)
    monkeypatch.setattr(_base_ai_beamlet, "list_local_models", lambda: [])
    monkeypatch.setattr(
        _base_ai_beamlet, "list_hub_models", lambda org, offline=False: [f"{org}/dosecalc-a-model"]
    )

    assert AIBeamletEngine.is_available("protons", None) == (True, "")
    assert "AIBeamlet" in get_available_engines("protons")


def test_nothing_on_disk_and_nothing_on_the_hub(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(_base_ai_beamlet, "list_local_models", lambda: [])
    monkeypatch.setattr(_base_ai_beamlet, "list_hub_models", lambda org, offline=False: [])

    available, msg = AIBeamletEngine.is_available("protons", None)
    assert available is False
    assert "no network" in msg
    assert "AIBeamlet" not in get_available_engines("protons")


def test_hub_offering_only_outcome_models_does_not_count(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(_base_ai_beamlet, "list_local_models", lambda: [])
    monkeypatch.setattr(
        _base_ai_beamlet, "list_hub_models", lambda org, offline=False: [f"{org}/outcome-a-model"]
    )

    available, _ = AIBeamletEngine.is_available("protons", None)
    assert available is False


def test_model_taken_from_prop_dose_calc():
    engine = AIBeamletEngine(_pln())
    assert engine.model == DUMMY_MODEL
    assert AIBeamletEngine(_pln(model=None)).model is None


def test_availability_reads_model_from_engine_instance(met_requirements):
    engine = AIBeamletEngine(_pln())

    class _PlnWithEngine:
        radiation_mode = "protons"
        prop_dose_calc = engine

    assert AIBeamletEngine.is_available(_PlnWithEngine(), None) == (True, "")


def test_failing_check_is_not_offered(monkeypatch):
    def _broken(pln, machine):
        raise RuntimeError("boom")

    monkeypatch.setattr(AIBeamletEngine, "is_available", staticmethod(_broken))
    engines = get_available_engines("protons")
    assert "AIBeamlet" not in engines
    assert "HongPB" in engines


def test_get_engine_reports_reason(monkeypatch):
    monkeypatch.setattr(_base_ai_beamlet, "model_facts", lambda model_dir: ({}, None, ["torch"]))
    with pytest.warns(UserWarning, match="torch"), pytest.raises(ValueError):
        # The fallback engine then rejects a plan naming a different engine.
        get_engine(_pln())


def test_import_without_ml_framework(tmp_path):
    """Importing the engines must not crash without any ML stack installed."""
    models_dir = Path(__file__).resolve().parents[1] / "data" / "ai_models"
    code = textwrap.dedent(
        f"""
        import sys
        for name in ("torch", "safetensors", "huggingface_hub"):
            sys.modules[name] = None  # makes `import name` raise ImportError

        from pyRadPlan._settings import get_settings
        get_settings().ai.modelhub_local_models_dir = r"{models_dir}"
        get_settings().ai.modelhub_offline = True

        from pyRadPlan.dose.engines import get_available_engines
        from pyRadPlan.dose.engines._factory import DOSE_ENGINES

        assert "AIBeamlet" in DOSE_ENGINES
        # the only model available declares torch, which is not importable here
        assert "AIBeamlet" not in get_available_engines("protons")
        assert "HongPB" in get_available_engines("protons")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr


def _model_declaring(tmp_path, requirements, repo_id="OtherOrg/dosecalc-other-framework"):
    """Copy the dummy model, declaring *requirements* in its metadata."""
    src = Path(__file__).resolve().parents[1] / "data" / "ai_models" / "DKFZ-RadOpt"
    model_dir = tmp_path / Path(repo_id)
    shutil.copytree(src / "dummy_dosecalc_model", model_dir)

    config_file = model_dir / "model_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["metadata"]["requirements"] = requirements
    config_file.write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def test_model_may_declare_any_framework(monkeypatch, tmp_path):
    """The engine checks what a model declares, not torch: an ONNX model is treated alike."""
    _model_declaring(tmp_path, ["onnxruntime-there-is-no-such-package"])
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", tmp_path)

    pln = _pln(model="OtherOrg/dosecalc-other-framework")
    available, msg = AIBeamletEngine.is_available(pln, None)
    assert available is False
    assert "onnxruntime-there-is-no-such-package" in msg

    # the same model, declaring a framework that is installed, is usable
    _model_declaring(tmp_path, ["numpy"], repo_id="OtherOrg/dosecalc-installed-framework")
    assert AIBeamletEngine.is_available(
        _pln(model="OtherOrg/dosecalc-installed-framework"), None
    ) == (True, "")


def test_requirements_key_is_documented_for_model_authors():
    template = (
        Path(pyRadPlan.__file__).parent / "data" / "ai_models" / "model_config.template.json"
    )
    metadata = json.loads(template.read_text(encoding="utf-8"))["metadata"]
    assert "requirements" in metadata
    assert "_requirements_help" in metadata


def test_engine_offered_without_torch_for_a_model_that_does_not_need_it(tmp_path):
    """A model needing no torch makes the engine available on an install without torch.

    The counterpart of ``test_import_without_ml_framework``: there the only model
    declares torch and the engine is withheld. The model's code is never imported
    to find out, only its declaration is read.
    """
    _model_declaring(tmp_path, ["numpy"])
    code = textwrap.dedent(
        f"""
        import sys
        for name in ("torch", "safetensors"):
            sys.modules[name] = None  # makes `import name` raise ImportError

        from pyRadPlan._settings import get_settings
        get_settings().ai.modelhub_local_models_dir = r"{tmp_path}"
        get_settings().ai.modelhub_offline = True

        from pyRadPlan.dose.engines import AIBeamletEngine, get_available_engines

        pln = {{"radiation_mode": "protons",
               "prop_dose_calc": {{"model": "OtherOrg/dosecalc-other-framework"}}}}
        assert AIBeamletEngine.is_available(pln, None) == (True, ""), AIBeamletEngine.is_available(
            pln, None
        )
        assert "AIBeamlet" in get_available_engines("protons")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr
