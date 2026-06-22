docs:
    rm -rf docs/_autosummary
    rm -rf docs/_build
    uv run --group docs --all-extras sphinx-build -M html docs docs/_build

test:
    uvx --with tox-uv tox

lint:
    uvx pre-commit run --all-files

bench:
    uv run --group bench --all-extras pytest benches/ -v

cli *args:
    uv run --package anu-ctlab-io-cli anu-ctlab-io {{args}}
