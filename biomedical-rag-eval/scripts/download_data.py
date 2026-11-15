"""Download the PubMedQA labeled subset (ori_pqal.json) directly from its
public GitHub repository (MIT licensed) -- no API key or credentialed
access required. Skips the download if the file is already present unless
--force is given.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import RAW_PATH  # noqa: E402

URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--out", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        size = args.out.stat().st_size
        print(f"{args.out} already exists ({size} bytes); use --force to re-download")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} -> {args.out}")
    with urllib.request.urlopen(URL, timeout=60) as resp, open(args.out, "wb") as f:
        f.write(resp.read())
    print(f"Done ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
