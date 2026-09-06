.PHONY: setup test lint reproduce clean-results

setup:
	uv sync --extra dev --frozen

test:
	uv run pytest

lint:
	uv run ruff check src/systematic_alpha tests

check: lint test

reproduce:
	uv run systematic-alpha run --config configs/research.yaml --download-if-missing --output results/reproduced

clean-results:
	python -m systematic_alpha.cli clean-results
