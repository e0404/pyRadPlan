# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pyRadPlan import __version__

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
]

autosummary_generate = True


jupytext_formats = "ipynb,py:percent"  # tells it to treat .py as notebooks
# nb_execution_mode = "cache"

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
