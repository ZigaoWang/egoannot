.PHONY: help install install-dev lint fmt typecheck test check clean \
        run-ingest run-curate run-frames run-annotate run-assemble run-export run-all \
        status dashboard

PY := python
UV := uv
PKG := egoannot

help:
	@echo "Targets:"
	@echo "  install       - Install runtime deps with uv"
	@echo "  install-dev   - Install runtime + dev deps and pre-commit hooks"
	@echo "  lint          - ruff check"
	@echo "  fmt           - ruff --fix + black"
	@echo "  typecheck     - mypy"
	@echo "  test          - pytest"
	@echo "  check         - lint + typecheck + test"
	@echo "  run-*         - Pipeline stage entrypoints"
	@echo "  dashboard     - Launch the Streamlit review UI"

install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy src

test:
	pytest

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

run-ingest:
	$(PY) -m $(PKG).cli ingest $(ARGS)

run-curate:
	$(PY) -m $(PKG).cli curate $(ARGS)

run-frames:
	$(PY) -m $(PKG).cli frames --all $(ARGS)

run-annotate:
	$(PY) -m $(PKG).cli annotate --all $(ARGS)

run-assemble:
	$(PY) -m $(PKG).cli assemble --all $(ARGS)

run-export:
	$(PY) -m $(PKG).cli export --jsonl $(ARGS)

run-all:
	$(PY) -m $(PKG).cli run --all $(ARGS)

status:
	$(PY) -m $(PKG).cli status

dashboard:
	$(PY) -m $(PKG).cli dashboard $(ARGS)
