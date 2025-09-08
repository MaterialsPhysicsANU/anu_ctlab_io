# Sphinx configuration for anu-ctlab-io docs
project = "anu-ctlab-io"
author = "MaterialsPhysicsANU"  # FIXME
release = "0.2.0"  # FIXME: Auto?
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
