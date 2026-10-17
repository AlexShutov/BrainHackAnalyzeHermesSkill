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
    a = ap.parse_args()

    qvec = None
    t0 = time.time()
    if not a.no_embed:
        try:
            from fastembed import TextEmbedding
            m = TextEmbedding(model_name=R.MODEL)
            qvec = list(m.embed([a.query]))[0]
        except Exception as e:
            print(f"# embed unavailable ({e.__class__.__name__}), BM25-only", file=sys.stderr)
    embed_ms = (time.time() - t0) * 1000

    t1 = time.time()
    res = R.search(a.query, top_k=a.top_k, min_score=a.min_score, qvec=qvec)
    print(json.dumps({"query": a.query[:120], "embed_ms": round(embed_ms, 1),
                      "total_ms": round((time.time() - t1) * 1000, 1), "results": res},
                     ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
