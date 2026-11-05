.PHONY: test integration_test lint format build

test:
	uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/

integration_test:
	BILLABLE=$(BILLABLE) uv run --group test --group test_integration pytest tests/integration_tests/

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .
	uv run --group lint --group test mypy

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

build:
	uv build
