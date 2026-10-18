#!/usr/bin/env python3
"""Build the psychology RAG index: devices.json + every book found in psychology/.

Book sources (git-ignored, copyright): psychology/*.fb2 (chunked by FB2 <p> with
<section><title> labels) and psychology/books/*.jsonl (paragraph records produced
by book_ingest_pdf.py). Near-duplicate chunks are dropped at build time: a chunk
whose embedding is >= PSYCH_RAG_DEDUPE (default 0.90) cosine to an already-kept
document adds no new knowledge, so only non-overlapping material merges in.

Run once at install; rebuild only when the corpus changes.
    python3 rag_build.py
"""
import glob, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as R

FB2 = sorted(glob.glob(os.path.join(R.PSY, "*.fb2")))
JSONL = sorted(glob.glob(os.path.join(R.PSY, "books", "*.jsonl")))


def pack(pieces, tag):
    """Pack paragraph-sized pieces into CHUNK_SIZE chunks with OVERLAP."""
    chunks, cur, cur_sec = [], "", ""
    for sec, p in pieces:
        if len(cur) + len(p) + 1 > R.CHUNK_SIZE:
            if cur:
                chunks.append((cur_sec, cur))
            cur = (cur[-R.OVERLAP:] + " " + p).strip() if R.OVERLAP else p
            cur_sec = sec
        else:
            cur = (cur + " " + p).strip()
            if not cur_sec:
                cur_sec = sec
    if cur:
        chunks.append((cur_sec, cur))
    return [(tag + ": " + s if tag else s, t) for s, t in chunks]


def fb2_chunks(path):
    if not os.path.exists(path):
        return []
    tag = re.split(r"\s+-\s+", os.path.basename(path))[0].strip()
    d = open(path, encoding="utf-8", errors="replace").read()
    # section title per paragraph: walk the XML in order
    cur_section = ""
    out = []
    for m in re.finditer(r"<section[^>]*>\s*<title>(.*?)</title>|<p>(.*?)</p>", d, re.S):
        if m.group(1):
            cur_section = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
            continue
        p = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(2))).strip()
        if len(p) < 40:
            continue
        out.append((cur_section or "—", p))
    return pack(out, tag)


def jsonl_chunks(path):
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if len(d.get("text", "")) >= 40:
            out.append((d.get("section") or "—", d["text"]))
    return pack(out, "")


def raw_books():
    """Book sources to ingest: raw files in psychology/ + encrypted corpus/reference_N.enc."""
    out = []
    for ext in (".fb2", ".pdf", ".epub", ".doc", ".txt"):
        out += sorted(glob.glob(os.path.join(R.PSY, "*" + ext)))
    out = [p for p in out if not p.endswith(".ingest.txt")]
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import books_secure as BS
        import unicodedata as _ud
        _stem = lambda p_: _ud.normalize("NFC", os.path.splitext(os.path.basename(p_))[0])
        shipped = {_stem(BS.real_name(int(n))) for n in BS._manifest()}
        out = [p for p in out if _stem(p) not in shipped]
        out += BS.all_references()
    except ImportError:
        pass  # corpus module absent -> raw files only
    return out


def ensure_jsonl():
    """Fresh clones carry encrypted books but not the git-ignored jsonl extracts —
    encrypt any newly dropped raw books, then run book_ingest.py for every corpus
    book that has no extract yet, so the books reach the index."""
    try:
        import books_secure as BS
        n = BS.encrypt_new_books()
        if n:
            print(f"  encrypted {n} new book(s)")
    except ImportError:
        pass
    if not raw_books():
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import book_ingest as BI
    os.makedirs(BI.BOOKS, exist_ok=True)
    for p in raw_books():
        slug = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(BI.BOOKS, slug + ".jsonl")
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(p):
            continue
        print(f"  ingesting: {slug}")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "book_ingest.py"), p], check=True)


def book_chunks():
    ensure_jsonl()
    # re-glob AFTER ensure_jsonl: fresh clones have no jsonl at import time.
    # Single ingestion path: every book (raw or encrypted) goes through
    # book_ingest.py -> books/*.jsonl; no direct fb2 parsing here any more.
    chunks = []
    for p in sorted(glob.glob(os.path.join(R.PSY, "books", "*.jsonl"))):
        chunks += jsonl_chunks(p)
    return chunks


def np_bytes(v):
    import numpy as np
    return np.asarray(v, dtype=np.float32).tobytes()


def main():
    os.makedirs(os.path.join(R.PSY, "rag"), exist_ok=True)
    con = R.open_db()
    con.execute("DELETE FROM docs"); con.execute("DELETE FROM vec")
    devices = []
    for name, dev in enumerate(json.load(open(R.DEVICES, encoding="utf-8"))["devices"]):
        label = dev["name"]
        text = f"{dev['family']}: {dev['definition']}. Формулировка: {dev.get('formula') or json.load(open(R.DEVICES, encoding='utf-8'))['formula']}. Пример: {dev['example']} (источник: {dev['source']})"
        devices.append(("device", label, text))
    chunks = book_chunks()
    docs = devices + [("chunk", s, t) for s, t in chunks]

    from fastembed import TextEmbedding
    t0 = time.time()
    model = TextEmbedding(model_name=R.MODEL)
    load = time.time() - t0
    t0 = time.time()
    vecs = list(model.embed([t for _, _, t in docs], batch_size=32))

    # merge, not append: drop chunks that duplicate an already-kept document
    import numpy as np
    thresh = float(os.environ.get("PSYCH_RAG_DEDUPE", "0.90"))
    kept, kept_vecs, dropped = [], [], 0
    mat = None
    for (kind, label, text), v in zip(docs, vecs):
        v = np.asarray(v, dtype=np.float32)
        if kind == "chunk" and mat is not None and mat.size:
            sims = mat @ (v / (np.linalg.norm(v) + 1e-9))
            if float(sims.max()) >= thresh:
                dropped += 1
                continue
        vn = v / (np.linalg.norm(v) + 1e-9)
        kept.append((kind, label, text))
        kept_vecs.append(vn)
        mat = np.stack(kept_vecs) if len(kept_vecs) > 1 else vn[None, :]
    con.executemany("INSERT INTO docs(kind, label, text) VALUES (?,?,?)", kept)
    con.executemany("INSERT INTO vec(id, blob) VALUES (?,?)",
                    [(i + 1, np_bytes(v)) for i, v in enumerate(kept_vecs)])
    con.commit()
    print(f"docs: {len(kept)} (devices {sum(1 for k,_,_ in kept if k=='device')}, "
          f"chunks {sum(1 for k,_,_ in kept if k=='chunk')}; duplicates dropped: {dropped})")
    print(f"model warm/cold load: {load:.1f} s | index build: {time.time()-t0:.1f} s | db: {R.DB_PATH}")


if __name__ == "__main__":
    main()
