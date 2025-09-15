docs:
    rm -rf docs/_autosummary
    rm -rf docs/_build
    uv run --group docs --all-extras sphinx-build -M html docs docs/_build

test:
    uvx --with tox-uv tox
