.PHONY: install install-dev install-llm data test lint retrieval generation generation-llm figure all clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-llm:
	pip install -r requirements-llm.txt

data:
	python3 scripts/download_data.py
	python3 scripts/prepare_data.py

test:
	pytest -v

lint:
	ruff check .

retrieval:
	python3 scripts/run_retrieval_eval.py

generation:
	python3 scripts/run_generation_eval.py

# Needs a GPU machine with Hugging Face Hub access (see README).
generation-llm:
	python3 scripts/run_generation_eval_llm.py

figure:
	python3 scripts/plot_classical_vs_llm.py

all: data retrieval generation figure
	@echo "Done. See results/tables/*.csv and results/figures/*.png"

clean:
	rm -rf data/raw data/processed .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
