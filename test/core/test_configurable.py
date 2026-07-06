"""Tests for the ConfigurableAlgorithm mixin and generated config models."""

from typing import Annotated, Any, ClassVar, Literal, Optional, Union

import numpy as np
import pytest
from pydantic import Field, ValidationError

from pyRadPlan.core import (
    AlgorithmConfig,
    AlgorithmParameterMetadata,
    ConfigurableAlgorithm,
    field_constraints,
)


class _NonConfigurableMixin:
    """Simulates ProgressReporter: annotations must not become parameters."""

    reporter_setting: float


class _BaseAlgorithm(ConfigurableAlgorithm, _NonConfigurableMixin):
    short_name: ClassVar[str] = "base"
    name: ClassVar[str] = "Base Algorithm"

    threshold: Annotated[float, Field(gt=0.0, description="A threshold")] = 1.0
    mode: Literal["fast", "accurate"] = "fast"
    flag: bool = True
    items: list[int] = [1, 2]
    legacy_attr: Union[str, dict]  # no default, like pre-migration annotations

    def __init__(self):
        self._init_config_defaults()

    @property
    def derived_value(self):
        return self.threshold * 2

    @derived_value.setter
    def derived_value(self, value):
        self.threshold = value / 2


class _SubAlgorithm(_BaseAlgorithm):
    short_name: ClassVar[str] = "sub"

    threshold: Annotated[float, Field(gt=0.0, le=10.0)] = 2.0  # override
    extra_param: Annotated[int, AlgorithmParameterMetadata(advanced=True)] = 7


class _UnschematizableAlgorithm(ConfigurableAlgorithm):
    weird: "np.ufunc" = np.add  # arbitrary type without schema support

    def __init__(self):
        self._init_config_defaults()


def test_model_generation_and_cache():
    model = _BaseAlgorithm.config_model()
    assert model is _BaseAlgorithm.config_model()
    assert model.__name__ == "_BaseAlgorithmConfig"
    assert issubclass(model, AlgorithmConfig)

    sub_model = _SubAlgorithm.config_model()
    assert sub_model is not model
    assert sub_model.__name__ == "_SubAlgorithmConfig"


def test_field_collection():
    fields = _BaseAlgorithm.config_model().model_fields
    assert set(fields) == {"threshold", "mode", "flag", "items", "legacy_attr"}
    # ClassVars, private attrs, and non-configurable-base annotations excluded
    assert "short_name" not in fields
    assert "reporter_setting" not in fields


def test_subclass_inherits_and_overrides():
    fields = _SubAlgorithm.config_model().model_fields
    assert "extra_param" in fields
    assert "threshold" in fields
    constraints = field_constraints(fields["threshold"])
    assert constraints["le"] == 10.0
    assert _SubAlgorithm.config_model().model_fields["threshold"].default == 2.0


def test_field_constraints_extraction():
    fields = _BaseAlgorithm.config_model().model_fields
    constraints = field_constraints(fields["threshold"])
    assert constraints["gt"] == 0.0
    assert constraints["description"] == "A threshold"

    meta = field_constraints(_SubAlgorithm.config_model().model_fields["extra_param"])
    assert meta["param_meta"].advanced is True


def test_init_defaults_and_mutable_isolation():
    a = _BaseAlgorithm()
    b = _BaseAlgorithm()
    assert a.threshold == 1.0
    assert a.legacy_attr is None  # no class default -> None
    a.items.append(3)
    assert b.items == [1, 2]


def test_init_defaults_do_not_clobber():
    class _PresetAlgorithm(_BaseAlgorithm):
        def __init__(self):
            self.threshold = 5.0
            super().__init__()

    assert _PresetAlgorithm().threshold == 5.0


def test_apply_config_valid_values():
    a = _BaseAlgorithm()
    a.apply_config({"threshold": 3.5, "mode": "accurate"})
    assert a.threshold == 3.5
    assert a.mode == "accurate"


def test_apply_config_camel_case_alias():
    a = _SubAlgorithm()
    a.apply_config({"extraParam": 42})
    assert a.extra_param == 42


def test_apply_config_unknown_key_warns_but_assigns():
    a = _BaseAlgorithm()
    with pytest.warns(UserWarning, match="not found"):
        a.apply_config({"nonexistent": 1})
    assert a.nonexistent == 1


def test_apply_config_unknown_key_property_setter():
    a = _BaseAlgorithm()
    a.apply_config({"derived_value": 8.0})
    assert a.threshold == 4.0


def test_apply_config_invalid_value_warns_and_assigns_raw():
    a = _BaseAlgorithm()
    with pytest.warns(UserWarning, match="failed validation"):
        a.apply_config({"threshold": -1.0, "flag": False})
    assert a.threshold == -1.0  # raw value assigned
    assert a.flag is False  # valid keys still validated and assigned


def test_apply_config_strict_raises():
    a = _BaseAlgorithm()
    with pytest.raises(ValidationError):
        a.apply_config({"threshold": -1.0}, strict=True)


def test_apply_config_from_model():
    model = _BaseAlgorithm.config_model()
    cfg = model(threshold=2.5)
    a = _BaseAlgorithm()
    a.apply_config(cfg)
    assert a.threshold == 2.5
    assert a.mode == "fast"  # untouched fields not applied


def test_get_config_roundtrip():
    a = _BaseAlgorithm()
    a.threshold = 4.0
    cfg = a.get_config()
    assert cfg.threshold == 4.0

    b = _BaseAlgorithm()
    b.apply_config(cfg)
    assert b.threshold == 4.0


def test_unschematizable_field_falls_back_to_any():
    model = _UnschematizableAlgorithm.config_model()
    assert "weird" in model.model_fields
    inst = _UnschematizableAlgorithm()
    assert inst.weird is np.add
    inst.apply_config({"weird": np.multiply})
    assert inst.weird is np.multiply


def test_property_backed_parameter():
    class _PropertyAlgorithm(ConfigurableAlgorithm):
        managed: Union[str, bool]

        def __init__(self):
            self.managed = False
            self._init_config_defaults()

        @property
        def managed(self):
            return self._managed

        @managed.setter
        def managed(self, value):
            self._managed = bool(value)

    fields = _PropertyAlgorithm.config_model().model_fields
    assert fields["managed"].default is None  # property object not used as default

    a = _PropertyAlgorithm()
    assert a.managed is False  # _init_config_defaults must not clobber via setter
    a.apply_config({"managed": "yes"})
    assert a.managed is True  # assignment goes through the setter


def test_union_with_dict_stays_dict():
    a = _BaseAlgorithm()
    a.apply_config({"legacy_attr": {"resolution": {"x": 1}}})
    assert isinstance(a.legacy_attr, dict)
    assert a.legacy_attr["resolution"] == {"x": 1}
