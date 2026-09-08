"""Download the PubMedQA labeled subset (ori_pqal.json) directly from its
public GitHub repository (MIT licensed) -- no API key or credentialed
access required.
"""
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ori_pqal.json"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} -> {OUT_PATH}")
    urllib.request.urlretrieve(URL, OUT_PATH)
    print(f"Done ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
