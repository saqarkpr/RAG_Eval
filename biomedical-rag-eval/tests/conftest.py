"""Make `src` importable when tests are run from anywhere (mirrors the
sys.path.insert pattern already used by every script in scripts/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
