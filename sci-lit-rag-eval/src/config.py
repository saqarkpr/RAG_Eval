"""Load configs/default.yaml and merge CLI overrides on top of it, so the
values that produced results/report.md live in exactly one place."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"

DEFAULTS = {
    "retriever": "bm25",
    "seed": 42,
    "top_k": 5,
    "n_splits": 5,
    "ece_bins": 10,
    "tfidf_lsa": {"n_components": 16},
    "data": {"corpus": "data/corpus.json", "qa_benchmark": "data/qa_benchmark.json"},
}


def load_config(path=None) -> dict:
    """Return DEFAULTS updated with the YAML file at `path` (or the project
    default config). Nested dicts are merged one level deep."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def resolve_data_paths(cfg: dict, root: Path = ROOT) -> tuple[Path, Path]:
    """Absolute (corpus, qa_benchmark) paths; relative entries are taken
    relative to the project root."""
    corpus = Path(cfg["data"]["corpus"])
    qa = Path(cfg["data"]["qa_benchmark"])
    return (
        corpus if corpus.is_absolute() else root / corpus,
        qa if qa.is_absolute() else root / qa,
    )
