#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/download_data.py
python3 scripts/prepare_data.py
python3 scripts/run_retrieval_eval.py
python3 scripts/run_generation_eval.py
echo "Done. See results/tables/*.csv and results/figures/*.png"
