.PHONY: install install-dev data test retrieval generation figure all clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

data:
	python3 scripts/download_data.py
	python3 scripts/prepare_data.py

test:
	pytest -v

retrieval:
	python3 scripts/run_retrieval_eval.py

generation:
	python3 scripts/run_generation_eval.py

figure:
	python3 scripts/plot_classical_vs_llm.py

all: data retrieval generation
	@echo "Done. See results/tables/*.csv and results/figures/*.png"

clean:
	rm -rf data/raw data/processed .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
