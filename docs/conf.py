import importlib.metadata

# Sphinx configuration for anu-ctlab-io docs
project = "anu-ctlab-io"
copyright = "2025, the Australian National University (ANU)"
author = "Materials Physics, ANU"  # Can only be a single author, so can't match pyproject.toml. Find actual authors there.
release = importlib.metadata.version("anu_ctlab_io")
extensions = [
    # 'myst_parser',
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]
html_theme = "sphinx_rtd_theme"
autosummary_generate = True
source_suffix = {
    # '.md': 'markdown',
    ".rst": "restructuredtext",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}
