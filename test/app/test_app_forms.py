from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from pyRadPlan.app._forms import FormValidationError, build_form, validate_values
from pyRadPlan.app._planform import ENGINE_FORM_EXCLUDE, form_model


class _Inner(BaseModel):
    method: str = "cg"
    tol: float = 1e-3


class _Model(BaseModel):
    flag: bool = True
    count: int = Field(default=3, ge=1, le=10, description="How many")
    scale: float = Field(default=0.5, gt=0.0)
    mode: Literal["a", "b"] = "a"
    auto: Union[Literal["auto"], bool] = "auto"
    method: Union[Literal["x", "y"], str] = "x"
    maybe: Optional[float] = None
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inner: _Inner = Field(default_factory=_Inner)
    blob: dict = Field(default_factory=dict)


def _fields(form):
    return {field["name"]: field for field in form["fields"]}


def test_descriptor_types_bounds_and_defaults():
    form = build_form(_Model, form_id="m", title="Configure M", values={"count": 5})
    fields = _fields(form)
    assert fields["flag"]["type"] == "bool"
    assert fields["count"] == {
        "name": "count",
        "label": "Count",
        "type": "int",
        "default": 3,
        "min": 1,
        "max": 10,
        "description": "How many",
    }
    assert fields["scale"]["exclusive_min"] == 0.0
    assert [c["value"] for c in fields["mode"]["choices"]] == ["a", "b"]
    assert [c["value"] for c in fields["auto"]["choices"]] == ["auto", True, False]
    assert fields["method"]["editable_choice"] is True
    assert fields["maybe"]["type"] == "float" and fields["maybe"]["nullable"]
    assert fields["shift"]["type"] == "number_list" and fields["shift"]["length"] == 3
    assert fields["inner"]["type"] == "group"
    assert [f["name"] for f in fields["inner"]["fields"]] == ["inner.method", "inner.tol"]
    assert fields["blob"]["type"] == "json"
    assert form["values"]["count"] == 5 and form["explicit"] == ["count"]


def test_invalid_initial_values_are_dropped():
    form = build_form(_Model, form_id="m", title="M", values={"count": 99, "flag": False})
    assert form["explicit"] == ["flag"]
    assert form["values"]["count"] == 3


def test_validate_values_keeps_nested_overrides_and_bool_choices():
    values = validate_values(_Model, {"auto": "true", "inner": {"tol": 0.1}, "count": 2})
    assert values == {"auto": True, "count": 2, "inner": {"tol": 0.1}}


def test_validate_values_reports_fields():
    with pytest.raises(FormValidationError) as info:
        validate_values(_Model, {"count": 0})
    assert info.value.details[0]["field"] == "count"
    with pytest.raises(FormValidationError) as info:
        validate_values(_Model, {"nope": 1})
    assert info.value.details == [{"field": "nope", "message": "unknown field"}]


def test_engine_form_hides_fields_the_plan_sets_elsewhere():
    model, title, exclude, _ = form_model("engine", "SVDPB", "photons")
    form = build_form(model, form_id="engine:SVDPB", title=title, exclude=exclude)
    names = {field["name"] for field in form["fields"]}
    assert title == "Configure SVD Pencil Beam" or title.startswith("Configure ")
    assert not names & ENGINE_FORM_EXCLUDE
    assert {"enable_dij_sampling", "geometric_lateral_cutoff"} <= names
    assert {"cube_wed", "hlut"}.isdisjoint(names)
