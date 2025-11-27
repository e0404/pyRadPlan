import pytest

from typing import Any
import numpy as np
from numpydantic import NDArray
import array_api_strict as xp

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.core.xp_utils import Array


class DummyModel(PyRadPlanBaseModel):
    value: int
    array: NDArray
    xp_array: Any
    nested: dict
    object_array: NDArray


@pytest.fixture
def dummy_instance():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "xp_array": xp.asarray([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([None, np.array([3, 4])], dtype=object),
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def another_dummy_instance():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "xp_array": xp.asarray([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([None, np.array([3, 4])], dtype=object),
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def different_dummy_instance_value():
    data = {
        "value": 11,
        "array": np.array([1, 2, 3]),
        "xp_array": xp.asarray([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([None, np.array([3, 4])], dtype=object),
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def different_dummy_instance_array():
    data = {
        "value": 11,
        "array": np.array([1, 2, 4]),
        "xp_array": xp.asarray([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([None, np.array([3, 4])], dtype=object),
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def different_dummy_instance_object_array():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "xp_array": xp.asarray([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([np.array([3, 4]), None], dtype=object),
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def different_dummy_instance_xp_array():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "xp_array": xp.asarray([1, 2, 3, 4]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([np.array([3, 4]), None], dtype=object),
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def different_dummy_instance_nested():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "xp_array": xp.asarray([1, 2, 3]),
        "nested": {"a": {"a_2": np.array([1, 2, 3])}, "b": 2},
        "object_array": np.array([np.array([3, 4]), None], dtype=object),
    }
    return DummyModel.model_validate(data)


def test_operator_equality(dummy_instance, another_dummy_instance):
    assert dummy_instance == another_dummy_instance


def test_operator_inequality(
    dummy_instance,
    different_dummy_instance_value,
    different_dummy_instance_array,
    different_dummy_instance_object_array,
    different_dummy_instance_xp_array,
    different_dummy_instance_nested,
):
    assert dummy_instance != different_dummy_instance_value
    assert dummy_instance != different_dummy_instance_array
    assert dummy_instance != different_dummy_instance_object_array
    assert dummy_instance != different_dummy_instance_xp_array
    assert dummy_instance != different_dummy_instance_nested
