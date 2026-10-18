#!/usr/bin/env python3
"""Ingest a PDF book into the psychology corpus (paragraph records, one JSON per line).

The PDF itself and the extracted records stay out of the repository (copyright):
output goes to psychology/books/<slug>.jsonl, which is git-ignored. rag_build.py
packs these paragraphs into chunks exactly like the FB2 ones, so all books go
through one index path.

Usage:
  python3 book_ingest_pdf.py <book.pdf> [--out FILE] [--source "Book title"]
"""
import argparse, json, os, re, sys
from util import read_lines

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(os.path.dirname(HERE), "psychology", "books")

HEAD_RE = re.compile(r"^\s*(Глава|Часть|Раздел)\s+([0-9]+|[А-ЯA-Z]\b|[А-Яа-яA-Za-z]+)\s*$", re.I)
SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def clean_page(text):
    lines = []
    for raw in (text or "").split("\n"):
        s = raw.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,4}", s):            # page number
            continue
        if s.count(".") > 8:                        # TOC dot leaders
            continue
        lines.append(s)
    return lines


def paragraphs(lines):
    """Group the page-flow into paragraph-sized pieces (~500 chars)."""
    flow = " ".join(lines)
    sents = [s.strip() for s in SENT_RE.split(flow) if s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > 600:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def sections(pages, source):
    """Walk pages in order, split on chapter headings, yield (section, paragraph)."""
    sec = source
    title_tail = ""
    buf = []

    def flush():
        for p in paragraphs(buf):
            if len(p) >= 40:
                yield sec, p

    for lines in pages:
        i = 0
        while i < len(lines):
            s = lines[i]
            if HEAD_RE.match(s):
                yield from flush()
                buf.clear()
                # "Глава N" alone: the title is the next line
                name = s.strip()
                if len(s) < 15 and i + 1 < len(lines) and len(lines[i + 1]) < 120:
                    name = f"{s} — {lines[i + 1]}"
                    i += 1
                elif title_tail and len(s) < 15:
                    name = f"{s} — {title_tail}"
                sec = f"{source}. {name}"
                i += 1
                continue
            if len(s) > 15 and re.match(r"^[А-ЯЁA-Z«]", s) and len(s) < 120:
                title_tail = s  # candidate title line standing right before content
            buf.append(s)
            i += 1
    yield from flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=None, help="short book label for chunk labels")
    a = ap.parse_args()

    from pypdf import PdfReader
    r = PdfReader(a.pdf)
    source = a.source or os.path.splitext(os.path.basename(a.pdf))[0]
    pages = [clean_page(p.extract_text()) for p in r.pages]

    out = a.out or os.path.join(BOOKS, os.path.splitext(os.path.basename(a.pdf))[0] + ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for sec, p in sections(pages, source):
            fh.write(json.dumps({"section": sec, "text": p}, ensure_ascii=False) + "\n")
            n += 1
    secs = {}
    for line in read_lines(out):
        d = json.loads(line)
        secs[d["section"]] = secs.get(d["section"], 0) + 1
    print(f"paragraphs: {n} -> {out}")
    for k, v in list(secs.items())[:20]:
        print(f"  {v:4d}  {k[:90]}")


if __name__ == "__main__":
    main()
