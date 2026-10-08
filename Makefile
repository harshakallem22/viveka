.DEFAULT_GOAL := help
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: help setup setup-data test test-all lint fmt typecheck report judge validate-judge gate baseline dashboard serve demo-db cassettes golden-set golden-set-check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package (editable) with dev extras
	$(PIP) install -e ".[dev]"

setup-data:  ## Additionally install the heavy HF datasets extra (only needed to rebuild the golden set)
	$(PIP) install -e ".[dev,data]"

test:  ## Run tests that do not need a live Ollama server
	$(PY) -m pytest -m "not ollama"

test-all:  ## Run every test, including ones that hit Ollama
	$(PY) -m pytest

lint:  ## Lint
	$(PY) -m ruff check .

fmt:  ## Auto-format and fix imports
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

typecheck:  ## Type-check the package
	$(PY) -m mypy

golden-set:  ## Rebuild the frozen golden set (requires the `data` extra)
	$(PY) scripts/build_golden_set.py

report:  ## Aggregate the latest run into a P50/P95 leaderboard + results/latest.json
	$(PY) -m viveka.cli report latest --json results/latest.json

golden-set-check:  ## Verify the committed golden set matches its recorded hash
	$(PY) -c "from viveka.core.datasets import load_golden_set; \
items = load_golden_set('data/golden_set.jsonl'); \
print(f'OK: {len(items)} items, hash verified')"

clean:  ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} +

judge:  ## Score the latest run with each item's designated grader
	$(PY) -m viveka.cli judge latest

validate-judge:  ## Measure judge reliability against objective ground truth
	$(PY) -m viveka.cli validate-judge latest --json results/judge_validation.json

gate:  ## Run the CI regression gate against the committed baseline
	$(PY) -m viveka.cli gate --results results/latest.json --baseline results/baseline.json

baseline:  ## Promote the current results to be the accepted baseline
	$(PY) -m viveka.cli gate --update-baseline

cassettes:  ## Re-record CI fixtures from the latest local run
	$(PY) scripts/record_cassettes.py --run latest

demo-db:  ## Build the committed read-only demo database for the hosted dashboard
	$(PY) scripts/make_demo_db.py --run latest

dashboard:  ## Build the React dashboard into dashboard/dist
	cd dashboard && npm install --silent && npm run build

serve: dashboard  ## Build the dashboard and serve API + UI on :8000
	$(PY) -m viveka.cli serve

serve-demo:  ## Serve the committed demo database (no Ollama needed)
	VIVEKA_DB=data/demo.db $(PY) -m viveka.cli serve
