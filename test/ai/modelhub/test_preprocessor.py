import pytest

from pyRadPlan.ai.modelhub import BasePreprocessor


def test_base_preprocessor_is_abstract():
    with pytest.raises(TypeError):
        BasePreprocessor()


def test_subclass_call_delegates_to_preprocess():
    class Doubler(BasePreprocessor):
        def preprocess(self, inputs):
            return inputs * 2

    pre = Doubler({"key": "value"})
    assert pre.config == {"key": "value"}
    assert pre(3) == 6
    assert pre.preprocess(4) == 8


def test_default_postprocess_is_identity():
    class Identity(BasePreprocessor):
        def preprocess(self, inputs):
            return inputs

    pre = Identity()
    assert pre.config == {}
    assert pre.postprocess(42) == 42
