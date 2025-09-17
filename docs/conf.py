import importlib.metadata

# Sphinx configuration for anu-ctlab-io docs
project = "anu-ctlab-io"
copyright = "2025, the Australian National University (ANU)"
author = "Materials Physics, ANU"  # Can only be a single author, so can't match pyproject.toml. Find actual authors there.
release = importlib.metadata.version("anu_ctlab_io")
extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "sphinx_autodoc_typehints",
]
html_theme = "sphinx_rtd_theme"
source_suffix = {
    ".rst": "restructuredtext",
}
autoapi_dirs = ["../src/anu_ctlab_io"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "imported-members"]
autodoc_typehints = "description"
exclude_patterns = ["autoapi/index.rst"]
