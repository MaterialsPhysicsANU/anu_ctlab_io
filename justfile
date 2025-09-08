docs:
    uv run sphinx-build -M html docs docs/_build

test:
    uv run pytest
