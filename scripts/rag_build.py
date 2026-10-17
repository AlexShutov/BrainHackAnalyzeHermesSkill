#!/usr/bin/env python3
"""Build the psychology RAG index: devices.json + the book, chunked by FB2 <p>.

Run once at install; rebuild only when the corpus changes.
    python3 rag_build.py
"""
import json, os, re, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as R

BOOK = os.path.join(R.PSY, "Спирица Е., Пелехатый М. - Боевое НЛП (Сам себе психолог Питер) - 2020.fb2")

def book_chunks():
    if not os.path.exists(BOOK):
        # the book is not shipped with the skill (copyright) — devices-only index is valid
        return []
    d = open(BOOK, encoding="utf-8", errors="replace").read()
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
        out.append((cur_section, p))
    chunks, cur, cur_sec = [], "", ""
    for sec, p in out:
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
    return chunks

def main():
    os.makedirs(os.path.join(R.PSY, "rag"), exist_ok=True)
    con = R.open_db()
    con.execute("DELETE FROM docs"); con.execute("DELETE FROM vec")
    docs = []
    for name, dev in enumerate(json.load(open(R.DEVICES, encoding="utf-8"))["devices"]):
        label = dev["name"]
        text = f"{dev['family']}: {dev['definition']}. Формулировка: {dev.get('formula') or json.load(open(R.DEVICES, encoding='utf-8'))['formula']}. Пример: {dev['example']} (источник: {dev['source']})"
        docs.append(("device", label, text))
    for sec, ch in book_chunks():
        docs.append(("chunk", sec or "—", ch))
    con.executemany("INSERT INTO docs(kind, label, text) VALUES (?,?,?)", docs)
    con.commit()

    from fastembed import TextEmbedding
    t0 = time.time()
    model = TextEmbedding(model_name=R.MODEL)
    load = time.time() - t0
    t0 = time.time()
    vecs = list(model.embed([t for _, _, t in docs], batch_size=32))
    con.executemany("INSERT INTO vec(id, blob) VALUES (?,?)",
                    [(i + 1, np_bytes(v)) for i, v in enumerate(vecs)])
    con.commit()
    print(f"docs: {len(docs)} (devices {sum(1 for k,_,_ in docs if k=='device')}, chunks {sum(1 for k,_,_ in docs if k=='chunk')})")
    print(f"model warm/cold load: {load:.1f} s | index build: {time.time()-t0:.1f} s | db: {R.DB_PATH}")

def np_bytes(v):
    import numpy as np
    return np.asarray(v, dtype=np.float32).tobytes()

if __name__ == "__main__":
    main()
