"""
PDF OCR -> Markdown using OpenAI Vision (gpt-4o-mini).

For scanned PDFs that have no text layer. Renders each page to PNG,
sends to OpenAI Vision, and stitches the markdown output.

Usage:
    python scripts/pdf_ocr_to_md.py
    python scripts/pdf_ocr_to_md.py --force          # re-OCR even if .md exists
    python scripts/pdf_ocr_to_md.py --model gpt-4o   # higher accuracy, ~10x cost
    python scripts/pdf_ocr_to_md.py --workers 8      # more parallelism

Cost estimate (gpt-4o-mini): ~$0.001/page. 41 pages ~= $0.04.
"""

import argparse
import base64
import glob
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, "data")
DEFAULT_DST = os.path.join(ROOT, "data")
load_dotenv(os.path.join(ROOT, ".env"))

PROMPT = (
    "Bạn là OCR engine cho tài liệu tiếng Việt. "
    "Trích xuất TOÀN BỘ văn bản trong ảnh thành Markdown sạch:\n"
    "- Giữ nguyên cấu trúc heading (#, ##, ###) dựa trên kích thước chữ.\n"
    "- Giữ nguyên bảng dùng cú pháp Markdown table.\n"
    "- Giữ nguyên danh sách (-, 1.).\n"
    "- KHÔNG thêm comment, KHÔNG bọc trong ```markdown```.\n"
    "- Nếu trang trống/không đọc được, trả về chuỗi rỗng.\n"
    "Chỉ trả về markdown, không gì khác."
)


def slugify(name: str) -> str:
    name = os.path.splitext(name)[0].lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def render_page_png(page: "pymupdf.Page", zoom: float = 2.0) -> bytes:
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def ocr_page(client: OpenAI, model: str, png_bytes: bytes, retries: int = 6) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        }},
                    ],
                }],
                max_tokens=4096,
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            is_429 = "429" in str(e) or "rate_limit" in str(e).lower()
            sleep_s = 30 + 10 * attempt if is_429 else 2 ** attempt
            time.sleep(sleep_s)
    raise RuntimeError(f"OCR failed after {retries} attempts: {last_err}")


def ocr_pdf(client: OpenAI, model: str, pdf_path: str, workers: int,
            page_indices: list[int] | None = None) -> dict[int, str]:
    """Render & OCR pages. Returns {page_index: markdown_text}."""
    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)
    if page_indices is None:
        page_indices = list(range(n_pages))
    pages_png = {i: render_page_png(doc[i]) for i in page_indices}
    doc.close()

    results: dict[int, str] = {}
    n = len(page_indices)
    print(f"  OCR'ing {n}/{n_pages} pages with {workers} workers...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ocr_page, client, model, png): i
                for i, png in pages_png.items()}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            done += 1
            try:
                results[i] = fut.result()
                print(f"    [{done}/{n}] page {i+1} OK ({len(results[i])} chars)", flush=True)
            except Exception as e:
                results[i] = f"<!-- OCR FAILED on page {i+1}: {e} -->"
                print(f"    [{done}/{n}] page {i+1} FAILED: {str(e)[:120]}", flush=True)

    elapsed = time.time() - start
    print(f"  OCR complete in {elapsed:.1f}s ({elapsed/max(n,1):.1f}s/page avg)")
    return results


def assemble_markdown(page_texts: dict[int, str], n_pages: int) -> str:
    parts = []
    for i in range(n_pages):
        text = page_texts.get(i, "").strip()
        if text:
            parts.append(f"<!-- page {i+1} -->\n\n{text}")
    return "\n\n---\n\n".join(parts)


def parse_existing_md(md_path: str, n_pages: int) -> tuple[dict[int, str], list[int]]:
    """Parse existing .md, return ({page_idx: text}, [failed_page_indices])."""
    if not os.path.exists(md_path):
        return {}, list(range(n_pages))
    txt = open(md_path, encoding="utf-8").read()
    sections = re.split(r"<!-- page (\d+) -->", txt)
    pages: dict[int, str] = {}
    for i in range(1, len(sections), 2):
        pno = int(sections[i])
        body = sections[i + 1].strip().rstrip("-").strip()
        pages[pno - 1] = body
    failed = [pno - 1 for pno in range(1, n_pages + 1)
              if pno - 1 not in pages
              or "OCR FAILED on page" in pages[pno - 1]
              or not pages[pno - 1]]
    return pages, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true",
                    help="re-OCR all pages even if .md exists")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-OCR only pages with 'OCR FAILED' marker in existing .md")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Create .env from .env.example.")
        return 1

    pdfs = sorted(glob.glob(os.path.join(args.src, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {args.src}")
        return 1

    os.makedirs(args.dst, exist_ok=True)
    client = OpenAI()
    print(f"OCR {len(pdfs)} PDF(s) with {args.model} -> {args.dst}/")

    for pdf in pdfs:
        out = os.path.join(args.dst, slugify(os.path.basename(pdf)) + ".md")
        doc = pymupdf.open(pdf)
        n_pages = len(doc)
        doc.close()

        existing, failed = parse_existing_md(out, n_pages)

        if args.force:
            todo = list(range(n_pages))
            existing = {}
        elif args.retry_failed:
            if not failed:
                print(f"  skip   {os.path.basename(pdf)} (no failed pages)")
                continue
            todo = failed
        else:
            if os.path.exists(out) and os.path.getsize(out) > 500 and not failed:
                print(f"  skip   {os.path.basename(pdf)} (exists, {os.path.getsize(out):,} bytes)")
                continue
            todo = failed if existing else list(range(n_pages))

        print(f"\n  >>>>>> {os.path.basename(pdf)}  ({len(todo)} pages to OCR)")
        try:
            new_results = ocr_pdf(client, args.model, pdf, args.workers, page_indices=todo)
            merged = {**existing, **new_results}
            md = assemble_markdown(merged, n_pages)
            with open(out, "w", encoding="utf-8") as f:
                f.write(md)
            still_failed = sum(1 for v in merged.values() if "OCR FAILED on page" in v)
            print(f"  saved  {os.path.basename(out)} ({len(md):,} chars, "
                  f"{still_failed} pages still FAILED)")
        except Exception as e:
            print(f"  FAIL   {os.path.basename(pdf)}: {e}")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
