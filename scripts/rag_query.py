#!/usr/bin/env python3
"""Query the psychology RAG. Prints compact results for the analysis skill.

    python3 rag_query.py "<text or key phrases>" [--top-k 6] [--no-embed]

--no-embed = BM25-only fallback (pure stdlib, ~0.3 ms) for machines where ONNX
cannot run; the skill still works, results are lexical-only.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=R.TOP_K)
    ap.add_argument("--min-score", type=float, default=R.MIN_SCORE)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--chunk", type=int, default=0,
                    help="делить длинный запрос на куски по N символов и опрашивать каждый "
                         "кусок отдельно (0 = один запрос). Без этого длинный текст попадает "
                         "в индекс только по первым ~512 токенам embedding-модели, и устройства "
                         "из хвоста текста не находятся никогда.")
    a = ap.parse_args()

    chunks = [a.query[i:i + a.chunk] for i in range(0, len(a.query), a.chunk)] if a.chunk else [a.query]

    qvecs = None
    t0 = time.time()
    if not a.no_embed:
        try:
            from fastembed import TextEmbedding
            m = TextEmbedding(model_name=R.MODEL)
            # модель загружается один раз, вектора — по всем кускам сразу
            qvecs = [list(m.embed([c]))[0] for c in chunks]
        except Exception as e:
            print(f"# embed unavailable ({e.__class__.__name__}), BM25-only", file=sys.stderr)
    embed_ms = (time.time() - t0) * 1000

    t1 = time.time()
    out = R.merge_chunked(chunks, top_k=a.top_k, min_score=a.min_score, qvecs=qvecs)
    print(json.dumps({"query": a.query[:120], "chunks": len(chunks),
                      "embed_ms": round(embed_ms, 1),
                      "total_ms": round((time.time() - t1) * 1000, 1), "results": out},
                     ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
