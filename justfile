docs:
    rm -rf docs/_autosummary
    rm -rf docs/_build
    uv run sphinx-build -M html docs docs/_build

test:
    uvx --with tox-uv tox
