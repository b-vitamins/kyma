.PHONY: bootstrap format lint typecheck test pre-commit

VENV_BIN := .venv/bin

bootstrap:
	bash scripts/bootstrap-venv.sh

format:
	$(VENV_BIN)/ruff format .

lint:
	$(VENV_BIN)/ruff check .

typecheck:
	$(VENV_BIN)/pyright

test:
	$(VENV_BIN)/pytest

pre-commit:
	bash scripts/pre-commit.sh
