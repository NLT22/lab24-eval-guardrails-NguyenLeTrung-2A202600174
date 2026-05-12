"""
PDF → Markdown converter (pre-processing step).

Convert all PDFs in data/ to markdown files so that load_documents()
in src/m1_chunking.py can pick them up.

Usage:
    python scripts/pdf_to_md.py
    python scripts/pdf_to_md.py --force      # overwrite existing .md
    python scripts/pdf_to_md.py --src data --dst data
"""

import argparse
import glob
import os
import re
import sys
import time

import pymupdf4llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, "data")
DEFAULT_DST = os.path.join(ROOT, "data")


def slugify(name: str) -> str:
    name = os.path.splitext(name)[0].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def convert(pdf_path: str, out_path: str) -> int:
    md = pymupdf4llm.to_markdown(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return len(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--force", action="store_true", help="overwrite existing .md")
    args = ap.parse_args()

    pdfs = sorted(glob.glob(os.path.join(args.src, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {args.src}")
        return 1

    os.makedirs(args.dst, exist_ok=True)
    print(f"Converting {len(pdfs)} PDF(s) -> {args.dst}/")

    for pdf in pdfs:
        out = os.path.join(args.dst, slugify(os.path.basename(pdf)) + ".md")
        if os.path.exists(out) and not args.force:
            print(f"  skip   {os.path.basename(pdf)} -> {os.path.basename(out)} (exists)")
            continue
        start = time.time()
        try:
            n = convert(pdf, out)
            print(f"  done   {os.path.basename(pdf)} -> {os.path.basename(out)}  "
                  f"({n:,} chars, {time.time()-start:.1f}s)")
        except Exception as e:
            print(f"  FAIL   {os.path.basename(pdf)}: {e}")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
