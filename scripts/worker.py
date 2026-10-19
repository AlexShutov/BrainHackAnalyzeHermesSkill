#!/usr/bin/env python3
"""Тёплый worker слоёв: держит embedding и ONNX-классификатор загруженными и отвечает
JSON-lines по stdin/stdout.

Плагин раньше спавнил subprocess на каждое сообщение и платил загрузку модели каждый раз.
Замер на этой машине (2026-10-19): холодный путь — retrieval 1.54 с + классификатор 1.17 с
+ маркеры 0.03 с = ~2.7 с на многосмысловое сообщение. Тёплый путь — классификатор 15 мс на
абзац, маркеры 8 мс, FTS5-поиск ~1 мс (cosine-матрица кэшируется здесь же). Worker сводит
сообщение к ~50-70 мс.

Протокол (одна JSON-строка на запрос, одна на ответ):
  {"op":"ping"}                                        -> {"ok":true,"warm_ms":...,"embedding":bool,"classifier":bool}
  {"op":"search","query":text,"chunks":[...]}          -> {"ok":true,"results":[...]}
  {"op":"markers","text":...}                          -> {"ok":true,"markers":[...]}
  {"op":"classify","text":...,"min":0.70,"gated":true} -> {"ok":true,"classification":[...]}

PSYCH_WORKER_NO_MODELS=1 — не грузить модели: поиск только лексический, классификатор
пустой. Это режим тестов и деградации, когда ONNX/fastembed недоступны.
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers, rag_core as R

NO_MODELS = os.environ.get("PSYCH_WORKER_NO_MODELS") == "1"
CHUNK = R.CHUNK


def load_embedding():
    if NO_MODELS:
        return None
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(model_name=R.MODEL)
    except Exception:
        return None


def load_classifier():
    if NO_MODELS:
        return None
    try:
        import classifier as C
        C.classify("прогрев")          # первая классификация платит загрузку ONNX
        return C
    except Exception:
        return None


def main():
    t0 = time.time()
    con = R.open_db()
    emb = load_embedding()
    cls = load_classifier()
    matrix = None

    def cached_matrix():
        nonlocal matrix
        if matrix is None and emb is not None:
            import numpy as np
            rows = list(con.execute("SELECT id, blob FROM vec"))
            if rows:
                ids = [r[0] for r in rows]
                e = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
                matrix = (ids, e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
        return matrix

    def emit(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"ok": True, "warm_ms": round((time.time() - t0) * 1000),
          "embedding": emb is not None, "classifier": cls is not None,
          "docs": con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            emit({"ok": False, "error": "bad json"})
            continue
        op = req.get("op")
        if op == "ping":
            emit({"ok": True})
        elif op == "search":
            q = req.get("query") or ""
            chunks = req.get("chunks") or [q]
            qvecs = [list(emb.embed([c]))[0] for c in chunks] if emb else [None] * len(chunks)
            emit({"ok": True, "results": R.merge_chunked(chunks, con=con, qvecs=qvecs,
                                                         matrix=cached_matrix())})
        elif op == "markers":
            emit({"ok": True, "markers": markers.scan(req.get("text") or "")})
        elif op == "classify":
            text = req.get("text") or ""
            minp = float(req.get("min", 0.70))
            if cls is None:
                emit({"ok": True, "classification": [], "model": None})
            elif req.get("gated"):
                res = []
                for i, p in cls.gated_paragraphs(text):
                    labels = cls.classify(p, min_conf=minp)
                    if labels:
                        res.append({"para": i, "text": p, "labels": labels})
                emit({"ok": True, "classification": res})
            else:
                emit({"ok": True, "classification": cls.classify(text, min_conf=minp)})
        else:
            emit({"ok": False, "error": f"unknown op: {op}"})
    con.close()


if __name__ == "__main__":
    main()
