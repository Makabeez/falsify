PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup test judge-demo data verify web clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"

# One command, no API key, no network: verifies bundled data hashes, runs every claim in claims/
judge-demo: setup
	$(BIN)/falsify judge-demo

test: setup
	$(BIN)/pytest -q

data: setup
	$(BIN)/falsify data fetch

# Web app on http://127.0.0.1:8097 (live checks need MODEL_API_KEY in .env)
web: setup
	./deploy/run-web.sh

verify:
	$(BIN)/falsify data verify

clean:
	rm -rf $(VENV) *.egg-info .pytest_cache
