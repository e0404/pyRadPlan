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

import jupytext
from sphinx.util import logging

from pyRadPlan import __version__

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
# The scripts in ../examples are jupytext "percent" notebooks. They are converted to .ipynb
# at build time (see setup() below) and rendered by myst-nb.
#
# Execution is OFF by default: several examples need external tools (MATLAB/Octave, FRED,
# TOPAS, a Qt display, AI credentials) and the remaining ones take far too long for a
# readthedocs build. Executed notebooks (with outputs) are committed in docs/tutorials/examples/;
# refresh them locally with `python docs/execute_examples.py` (never in CI). Scripts without a
# committed notebook are converted at build time without outputs (see setup() below).
# Set PYRADPLAN_DOCS_NB_EXECUTION=force (or auto/cache) to execute during a local build instead.
jupytext_formats = "ipynb,py:percent"
nb_execution_mode = os.environ.get("PYRADPLAN_DOCS_NB_EXECUTION", "off")
nb_execution_timeout = -1  # No per-cell timeout when execution is enabled
nb_execution_raise_on_error = False
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
DOCS_DIR = pathlib.Path(__file__).resolve().parent
EXAMPLES_DIR = DOCS_DIR.parent / "examples"
NOTEBOOKS_DIR = DOCS_DIR / "tutorials" / "examples"
NOTEBOOK_METADATA = {  # lets myst-nb pick the python lexer for notebooks converted without execution
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
}


def generate_example_notebooks(app):
    """Convert example scripts without a committed (executed) notebook to output-less notebooks."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for py_file in sorted(EXAMPLES_DIR.glob("*.py")):
        target = NOTEBOOKS_DIR / f"{py_file.stem}.ipynb"
        if not target.is_file():
            nb = jupytext.read(py_file)
            nb.metadata.update(NOTEBOOK_METADATA)
            jupytext.write(nb, target)
            generated.append(target.name)
    if generated:
        logger.info(
            "Generated %d example notebooks without outputs: %s", len(generated), generated
        )


def cleanup_jupyter_execute(app, exception):
    folder = os.path.join(app.srcdir, "jupyter_execute")
    if os.path.exists(folder):
        shutil.rmtree(folder)
        logger.info("Removed jupyter_execute folder after build")


def setup(app):
    app.connect("builder-inited", generate_example_notebooks)
    app.connect("build-finished", cleanup_jupyter_execute)
