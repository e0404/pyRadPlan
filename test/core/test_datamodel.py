import pytest
import numpy as np
import numpy as np

from pyRadPlan.core import PyRadPlanBaseModel


class DummyModel(PyRadPlanBaseModel):
    value: int
    array: list[int]
    nested: dict


@pytest.fixture
def dummy_instance():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def another_dummy_instance():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 2, 3])}, "b": 2},
    }
    return DummyModel.model_validate(data)


@pytest.fixture
def different_dummy_instance():
    data = {
        "value": 10,
        "array": np.array([1, 2, 3]),
        "nested": {"a": {"a_1": np.array([1, 3, 2])}, "b": 2},
    }
    return DummyModel.model_validate(data)


def test_operator_equality(dummy_instance, another_dummy_instance):
    assert dummy_instance == another_dummy_instance


def test_operator_inequality(dummy_instance, different_dummy_instance):
    assert dummy_instance != different_dummy_instance
