#!/usr/bin/env python3
"""Core of the psychology RAG: SQLite (FTS5 lexical) + vectors, hybrid scoring.

Shared by rag_build.py, rag_query.py and rag_mcp.py so parameters are identical
everywhere. No torch, no ANN: the corpus is book-sized, brute-force cosine is
sub-millisecond.
"""
import json, math, os, re, sqlite3

import side   # боковой канал для укр/бел — отдельный, индекс не перестраивается

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSY = os.path.join(SKILL_DIR, "psychology")
DB_PATH = os.environ.get("PSYCH_RAG_DB", os.path.join(PSY, "rag", "psych.db"))
DEVICES = os.path.join(PSY, "devices.json")

# Parameters (fixed, recorded in the skill)
CHUNK_SIZE = 1200
OVERLAP = 200
TOP_K = 6
MIN_SCORE = 0.30
W_BM25, W_COS = 0.4, 0.6
NAME_BOOST = 0.15
MAX_OUT_CHARS = 2500
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def norm_words(text):
    return re.findall(r"[a-zа-яё]{3,}", text.lower())

def q_tokens(query):
    """Токены запроса: основной язык без изменений + боковые слова, приведённые к
    русской форме. Индекс не перестраивается: боковой язык не попадает в корпус, он
    переводится только на стороне запроса — так запрос на мове попадает в существующий
    русский индекс. Без этого lexical-половина для мовы даёт ноль совпадений."""
    toks = norm_words(query)
    for w in side.side_words(query)[0]:
        if w not in toks:
            toks.append(w)
    return toks

def open_db(readonly=True):
    """Запросная часть обязана открывать БД только для чтения: путь построения индекса
    держит write-лок, и обычное connect() на чтении ловит 'database is locked'.
    CREATE TABLE остаётся только в пути построения (там, где есть право на запись)."""
    if readonly and os.path.exists(DB_PATH):
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)   # чистая установка: каталога rag/ ещё нет
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, kind TEXT, label TEXT, text TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS vec(id INTEGER PRIMARY KEY REFERENCES docs(id), blob BLOB)")
    return con

def ensure_fts(con):
    """Настоящая FTS5-таблица. Раньше навык описывался как FTS5, но индекса в БД не было:
    лексическая половина была ручным полным проходом по корпусу (322 мс на запрос при
    9062 докуах). FTS5-запрос по тому же корпусу — 0-1 мс, то есть ~300x.

    Индексируются ДВЕ колонки: label и text. Прежний проход сканировал только text,
    поэтому запрос по имени устройства (например «ложная дилемма») находил определение
    лишь тогда, когда имя случайно встречалось в определении; отдельное имя устройства
    не находилось никогда. Колонка label с весом 2.0 — это recall для таких запросов.

    Индекс строится в пути построения (там есть право на запись); запросная часть
    работает по нему через mode=ro."""
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(label, text, content='docs', content_rowid='id')")


def fts_ready(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='docs_fts'").fetchone() is not None


def bm25_scores(con, query, k1=1.5, b=0.75, candidate_limit=200):
    """Лексическая половина: FTS5 bm25(), если индекс есть; иначе — прежний полный проход
    (старые БД без FTS5 и тесты на памяти). Нормализация по максимуму сохранена, чтобы
    веса гибрида (W_BM25/W_COS) и MIN_SCORE остались сравнимыми с прежними замерами."""
    q = q_tokens(query)
    if not q:
        return {}
    if fts_ready(con):
        rows = con.execute(
            "SELECT rowid, bm25(docs_fts, 2.0, 1.0) FROM docs_fts WHERE docs_fts MATCH ? "
            "ORDER BY bm25(docs_fts, 2.0, 1.0) LIMIT ?", (" OR ".join(q), candidate_limit)).fetchall()
        if not rows:
            return {}
        raw = {r[0]: -r[1] for r in rows}          # FTS5 отдаёт отрицательный rank: меньше = лучше
        mx = max(raw.values())
        if mx <= 0:
            return {}
        return {k: v / mx for k, v in raw.items()}
    rows = list(con.execute("SELECT id, text FROM docs"))
    N = len(rows)
    if not N:
        return {}
    toks = [norm_words(t) for _, t in rows]
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    avg = sum(len(t) for t in toks) / N
    scores = {}
    for w in q:
        if w not in df:
            continue
        idf = math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1)
        for i, (_, text) in enumerate(rows):
            f = toks[i].count(w)
            if f:
                s = idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks[i]) / avg))
                scores[rows[i][0]] = scores.get(rows[i][0], 0.0) + s
    if not scores:
        return {}
    mx = max(scores.values())
    return {k: v / mx for k, v in scores.items()}

def cosine_scores(con, qvec, matrix=None):
    import numpy as np
    if qvec is None:
        return {}
    if matrix is not None:
        ids, emb = matrix
    else:
        rows = list(con.execute("SELECT id, blob FROM vec"))
        if not rows:
            return {}
        ids = [r[0] for r in rows]
        emb = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    s = emb @ q
    return {i: float(v) for i, v in zip(ids, s)}

def snippet(text, limit=180):
    s = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", s)
    out = ""
    for p in parts[:2]:
        if len(out) + len(p) > limit:
            break
        out += p + " "
    return out.strip() or s[:limit]

def search(query, top_k=TOP_K, min_score=MIN_SCORE, con=None, qvec=None, matrix=None):
    own = con is None
    con = con or open_db()
    lex = bm25_scores(con, query)
    cos = cosine_scores(con, qvec, matrix) if qvec is not None else {}
    ql = q_tokens(query)
    scored = {}
    for doc_id, s in lex.items():
        scored[doc_id] = W_BM25 * s
    for doc_id, s in cos.items():
        scored[doc_id] = scored.get(doc_id, 0.0) + W_COS * s
    # один запрос на все кандидатов вместо SELECT на каждый: раньше boost-цикл и резерв
    # устройств делали по запросу на документ (до 200 запросов на запрос)
    ids = list(scored)
    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute(
        "SELECT id, kind, label, text FROM docs WHERE id IN (" + ",".join("?" * len(ids)) + ")", ids)} if ids else {}
    # exact device-name boost from the lexical side
    for doc_id, s in scored.items():
        row = meta.get(doc_id)
        if row and row[0] == "device" and any(w in norm_words(row[1]) for w in ql):
            scored[doc_id] = s + NAME_BOOST
    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    out = []
    total = 0
    for doc_id, score in ranked:
        if score < min_score or len(out) >= top_k:
            break
        kind, label, text = meta[doc_id]
        sn = snippet(text)
        cost = len(sn) + len(label) + 20
        if total + cost > MAX_OUT_CHARS:
            break
        total += cost
        out.append({"id": doc_id, "kind": kind, "label": label, "score": round(score, 3), "snippet": sn})
    # a big corpus lets prose chunks crowd device definitions out of top-k; the
    # definitions are the point of retrieval, so reserve slots for the two best
    # devices above a floor even when chunks outrank them
    if sum(1 for r in out if r["kind"] == "device") < 2:
        have = {r["id"] for r in out}
        spare = [(doc_id, score) for doc_id, score in ranked
                 if doc_id not in have and score >= 0.20 and meta.get(doc_id, ("",))[0] == "device"]
        for doc_id, score in spare[: 2 - sum(1 for r in out if r["kind"] == "device")]:
            kind, label, text = meta[doc_id]
            sn = snippet(text)
            if total + len(sn) + len(label) + 20 > MAX_OUT_CHARS:
                break
            total += len(sn) + len(label) + 20
            out.append({"id": doc_id, "kind": kind, "label": label, "score": round(score, 3), "snippet": sn})
    if own:
        con.close()
    return out


CHUNK = 1500

def chunks_of(text, size=CHUNK):
    return [text[i:i + size] for i in range(0, len(text), size)] if size else [text]


def merge_chunked(chunks, top_k=TOP_K, min_score=MIN_SCORE, qvecs=None, con=None, matrix=None):
    """Один запрос на кусок: embedding видит только первые ~512 токена, и устройства из
    хвоста длинного текста без кусков не находятся никогда. Слитый результат — лучший hit
    по каждому устройству.

    Бюджет MAX_OUT_CHARS применяется к слитому выводу, а не к каждому куску: раньше при
    трёх кусках вывод мог быть втрое больше обещанных 2500 символов — лимит был на кусок.
    `matrix` — кэш нормализованной матрицы векторов (тёплый worker); без него матрица
    пересобирается на каждый запрос (~10 мс на 9k докуах).
    """
    merged = {}
    for i, c in enumerate(chunks):
        for r in search(c, top_k=top_k, min_score=min_score, con=con,
                        qvec=(qvecs[i] if qvecs else None), matrix=matrix):
            key = (r["kind"], r["label"])
            if key not in merged or r["score"] > merged[key]["score"]:
                merged[key] = dict(r)
                merged[key]["chunk"] = i + 1
    ranked = sorted(merged.values(), key=lambda r: -r["score"])
    out, total = [], 0
    for r in ranked:
        if len(out) >= top_k:
            break
        cost = len(r["snippet"]) + len(r["label"]) + 20
        if total + cost > MAX_OUT_CHARS:
            break
        total += cost
        out.append(r)
    return out
