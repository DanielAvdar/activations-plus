.PHONY: help
.PHONY: default
default: install


install:
	uv run pre-commit install
	uv sync --all-extras --all-groups --frozen
	uv pip install pre-commit

install-docs:
	uv sync --group docs --frozen --no-group dev

update:
	uv lock
	uvx pre-commit autoupdate
	$(MAKE) install

test:
	uv run pytest

check:
	uv run pre-commit run --all-files

cov:
	uv run pytest --cov=activations_plus --cov-report=term-missing


coverage:
	uv run pytest --cov=activations_plus --cov-report=xml

mypy:
	uv  run mypy activations_plus --config-file pyproject.toml

doc:
	uv run sphinx-build -M html docs/source docs/build/
