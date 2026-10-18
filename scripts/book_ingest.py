#!/usr/bin/env python3
"""Ingest a book (PDF, EPUB, .doc/.txt) into the psychology corpus.

Output: psychology/books/<slug>.jsonl — one paragraph per line, git-ignored
(copyright). rag_build.py packs these paragraphs into chunks exactly like the
FB2 ones, so every book goes through one index path.

Usage:
  python3 book_ingest.py <book.pdf>  [--source "Автор «Название»"]
  python3 book_ingest.py <book.epub> [...]
  python3 book_ingest.py <book.doc|.txt> [...]
  python3 book_ingest.py status          # что уже в corpus
"""
import argparse, json, os, re, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(os.path.dirname(HERE), "psychology", "books")

HEAD_RE = re.compile(r"^\s*(Глава|Часть|Раздел)\s+([0-9]+|[А-ЯA-Z]\b|[А-Яа-яA-Za-z]+)\s*$", re.I)
SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def clean_lines(lines):
    out = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if s.count(".") > 8:
            continue
        out.append(s)
    return out


def paragraphs(lines, size=600):
    flow = " ".join(lines)
    sents = [s.strip() for s in SENT_RE.split(flow) if s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > size:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def sections(pages_lines, source):
    """pages_lines: список страниц/глав, каждая — список строк. Сплит по заголовкам глав."""
    sec = source
    title_tail, buf = "", []
    for lines in pages_lines:
        i = 0
        while i < len(lines):
            s = lines[i]
            if HEAD_RE.match(s):
                if buf:
                    for p in paragraphs(buf):
                        if len(p) >= 40:
                            yield sec, p
                    buf = []
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
                title_tail = s
            buf.append(s)
            i += 1
    for p in paragraphs(buf):
        if len(p) >= 40:
            yield sec, p


def pdf_pages(path):
    from pypdf import PdfReader
    r = PdfReader(path)
    return [clean_lines((p.extract_text() or "").split("\n")) for p in r.pages]


def epub_pages(path):
    """EPUB: порядок файлов из spine, текст из xhtml, секция = первый h1-h3 файла."""
    import xml.etree.ElementTree as ET
    pages = []
    with zipfile.ZipFile(path) as z:
        container = z.read("META-INF/container.xml").decode("utf-8", "replace")
        opf_path = re.search(r'full-path="([^"]+)"', container).group(1)
        opf = z.read(opf_path).decode("utf-8", "replace")
        base = os.path.dirname(opf_path)
        id2href = {}
        for m in re.finditer(r"<item\b[^>]*>", opf):
            tag = m.group(0)
            mid = re.search(r'id="([^"]+)"', tag)
            mhref = re.search(r'href="([^"]+)"', tag)
            if mid and mhref:
                id2href[mid.group(1)] = mhref.group(1)
        spine = re.findall(r'<itemref[^>]*idref="([^"]+)"', opf)
        for idref in spine:
            href = id2href.get(idref)
            if not href:
                continue
            full = os.path.join(base, href)
            try:
                html = z.read(full).decode("utf-8", "replace")
            except KeyError:
                continue
            title_m = re.search(r"<title>(.*?)</title>", html, re.S)
            head_m = re.search(r"<h[123][^>]*>(.*?)</h[123]>", html, re.S)
            title = re.sub(r"<[^>]+>", "", head_m.group(1) if head_m else (title_m.group(1) if title_m else "")).strip()
            body = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
            text = re.sub(r"<[^>]+>", "\n", body)
            text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                        .replace("&laquo;", "«").replace("&raquo;", "»")
                        .replace("&mdash;", "—").replace("&quot;", '"'))
            lines = clean_lines(text.split("\n"))
            if len(lines) < 3:
                continue
            if title:
                lines.insert(0, f"### {title}")
            pages.append(lines)
    return pages


def doc_or_txt_lines(path):
    if path.lower().endswith(".doc"):
        out = path + ".ingest.txt"
        res = subprocess.run(["textutil", "-convert", "txt", "-output", out, path],
                             capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"textutil: {res.stderr}")
        src = out
    else:
        src = path
    return clean_lines(open(src, encoding="utf-8", errors="replace").read().split("\n"))


def ingest(path, source, out_path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = pdf_pages(path)
    elif ext == ".epub":
        pages = epub_pages(path)
    elif ext in (".doc", ".txt"):
        pages = [doc_or_txt_lines(path)]
    else:
        sys.exit(f"неподдерживаемый формат: {ext}")

    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for sec, para in sections(pages, source):
            fh.write(json.dumps({"section": sec, "text": para}, ensure_ascii=False) + "\n")
            n += 1
    secs = {}
    for line in open(out_path, encoding="utf-8"):
        secs[json.loads(line)["section"]] = secs.get(json.loads(line)["section"], 0) + 1
    print(f"paragraphs: {n} -> {out_path}")
    for k, v in list(secs.items())[:12]:
        print(f"  {v:4d}  {k[:90]}")


def status():
    for f in sorted(os.listdir(BOOKS)) if os.path.isdir(BOOKS) else []:
        n = sum(1 for _ in open(os.path.join(BOOKS, f), encoding="utf-8"))
        print(f"  {n:5d}  {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="book.pdf / .epub / .doc / .txt")
    ap.add_argument("--out")
    ap.add_argument("--source", help="короткая метка книги для заголовков чанков")
    ap.add_argument("status_", nargs="?", help="напиши 'status' вместо пути")
    a = ap.parse_args()
    if not a.path or a.path == "status":
        status()
        return
    slug = os.path.splitext(os.path.basename(a.path))[0]
    source = a.source or slug
    out = a.out or os.path.join(BOOKS, slug + ".jsonl")
    ingest(a.path, source, out)


if __name__ == "__main__":
    main()
