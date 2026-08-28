PY := .venv/bin/python

.PHONY: setup data train evaluate explain predict all test lint app clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m credit_risk.main generate-data

train:
	$(PY) -m credit_risk.main train

evaluate:
	$(PY) -m credit_risk.main evaluate

explain:
	$(PY) -m credit_risk.main explain

predict:
	$(PY) -m credit_risk.main predict

all: data train evaluate explain

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

app:
	.venv/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf data/*.csv models/*.joblib reports/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
