# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import pathlib
import shutil
import sys

from sphinx.util import logging

from pyRadPlan import __version__

DOCS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR))
import execute_examples  # noqa: E402 - shared with the standalone example runner

logger = logging.getLogger(__name__)

print(os.path.abspath("../pyRadPlan"))
sys.path.insert(0, os.path.abspath("../pyRadPlan"))  # Adjust to your source folder

project = "pyRadPlan"
copyright = "2024, e0404"
author = "e0404"

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_design",
    "myst_nb",  # For Jupyter notebooks
]

autosummary_generate = True

# -- Example notebooks (myst-nb) ---------------------------------------------
# The scripts in ../examples (jupytext percent format) are the source of the examples. When the
# build starts (see setup() below) they are converted to notebooks without outputs in
# tutorials/examples/ (ignored by git), and nothing is executed: neither Read the Docs nor CI run
# the examples. Release commits instead carry executed notebooks there, written by
# `python docs/execute_examples.py --release`; they are rendered as is while their script is
# unchanged (they opt out of execution in their metadata), and replaced by a conversion without
# outputs, with a warning, once their script changed. PYRADPLAN_DOCS_NB_EXECUTION=cache is a
# local opt-in to execute the notebooks without outputs with myst-nb (cached in
# _build/.jupyter_cache); examples needing tools a build system lacks (Qt display, FRED/TOPAS,
# MATLAB/Octave, AI credentials, see execute_examples.SKIP) are never executed.
execute_examples.set_example_environment()  # kernels inherit it: no GUI, inline figures
nb_execution_mode = os.environ.get("PYRADPLAN_DOCS_NB_EXECUTION", "off")
nb_execution_cache_path = str(DOCS_DIR / "_build" / ".jupyter_cache")
nb_execution_excludepatterns = [
    f"tutorials/examples/{pattern}.ipynb" for pattern in execute_examples.SKIP
]
nb_execution_timeout = 1800  # per cell; guards the build against a hanging example
nb_execution_raise_on_error = False  # a failing example is reported, the build goes on
nb_execution_show_tb = True
nb_output_stderr = "remove"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

autodoc_type_aliases = {
    "npt.ArrayLike": "npt.ArrayLike",
}

autodoc_default_options = {
    "no-module": True,  # Suppress the module/package labels
}

numpydoc_class_members_toctree = False

# disable show json as it otherwises crashes the documentation building at the moment
autodoc_pydantic_model_show_json = False

# disable config summaries for inner classes
autodoc_pydantic_model_show_config_summary = False

# -- Example notebook generation ---------------------------------------------


def generate_example_notebooks(app):
    """Convert the example scripts to the notebooks of the tutorials section."""
    written, stale = execute_examples.write_notebooks()
    if written:
        logger.info("Converted %d example scripts to notebooks: %s", len(written), written)
    for name in stale:
        logger.warning(
            "The executed notebook tutorials/examples/%s is older than its script and is shown "
            "without outputs; rerun `python docs/execute_examples.py --release %s` on the "
            "release branch.",
            name,
            name.removesuffix(".ipynb"),
        )


def cleanup_jupyter_execute(app, exception):
    folder = os.path.join(app.srcdir, "jupyter_execute")
    if os.path.exists(folder):
        shutil.rmtree(folder)
        logger.info("Removed jupyter_execute folder after build")


def setup(app):
    app.connect("builder-inited", generate_example_notebooks)
    app.connect("build-finished", cleanup_jupyter_execute)
