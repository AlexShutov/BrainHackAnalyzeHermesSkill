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
import glob, json, os, re, sys, time
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


def book_chunks():
    chunks = []
    for p in FB2:
        chunks += fb2_chunks(p)
    for p in JSONL:
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
