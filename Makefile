.PHONY: format lint typecheck test pre-commit

format:
	ruff format .

lint:
	ruff check .

typecheck:
	pyright

test:
	pytest

pre-commit:
	bash scripts/pre-commit.sh
