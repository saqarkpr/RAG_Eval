#!/usr/bin/env bash
# Run the full pipeline (both retrievers) and produce the comparison report.
set -euo pipefail
cd "$(dirname "$0")/.."

python train.py --retriever bm25 --seed 42
python train.py --retriever tfidf_lsa --seed 42
python evaluate.py
