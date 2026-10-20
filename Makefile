.PHONY: install lint format typecheck test check

install:  ## venv + git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

test:
	uv run pytest

check: lint typecheck test  ## everything CI runs
