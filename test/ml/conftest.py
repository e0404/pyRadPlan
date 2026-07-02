import sys

import pytest

if sys.version_info < (3, 10):
    import importlib_resources as resources
else:
    from importlib import resources


@pytest.fixture
def example_model_dir():
    """Path to the shipped example model folder."""
    return str(resources.files("pyRadPlan.data.ml_models").joinpath("example_model"))
