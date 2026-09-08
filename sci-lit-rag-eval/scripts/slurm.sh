#!/usr/bin/env bash
#SBATCH --job-name=sci-lit-rag-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#
# This pipeline is intentionally CPU-only (BM25 / TF-IDF-LSA retrieval +
# a small scikit-learn classifier cascade, no neural generator), so it
# requests a small CPU allocation rather than a GPU. The full run
# (both retrievers, 5-fold CV, evaluation, plots) finishes in well under
# a minute on a single core; the SLURM wrapper exists to demonstrate a
# reproducible cluster-submission convention, not because the workload
# needs cluster-scale resources.

set -euo pipefail
cd "$(dirname "$0")/.."

module purge 2>/dev/null || true
# module load python/3.11   # uncomment / adjust for your cluster's module system

python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install --quiet -r requirements.txt

bash scripts/train.sh
