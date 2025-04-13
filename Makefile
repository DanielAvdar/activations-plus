.PHONY: help
.PHONY: default
default: install


install:
	uv sync --all-extras --all-groups --frozen
	uv tool install pre-commit --with pre-commit-uv --force-reinstall

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

coverage:
	uv run pytest --cov=activations_plus --cov-report=xml

mypy:
	uv tool run mypy activations_plus --config-file pyproject.toml

doc:
	uv run sphinx-build -M html docs/source docs/build/
