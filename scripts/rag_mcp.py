#!/usr/bin/env python3
"""MCP stdio server for the psychology RAG: one tool, `psychology_search`.

Long-lived process: the embedding model is loaded once (~0.5 s warm), then each
call is ~5-20 ms. Hermes config:

  mcp_servers:
    psychology-rag:
      command: /Users/alex/.hermes/tools/psych-rag-env/bin/python
      args: ["/Users/alex/.hermes/skills/brain-hack-analyze/scripts/rag_mcp.py"]
      enabled: true

BM25-only fallback if ONNX cannot load — the tool still answers, lexical only.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as R

PROTOCOL = "2024-11-05"
TOOL = {
    "name": "psychology_search",
    "description": ("Retrieval over the psychology corpus (closed list of manipulation devices + "
                    "the book). Call once per message analysis, before marking; pass the message text "
                    "or 3-6 key phrases. Returns top-k compact passages."),
    "inputSchema": {"type": "object", "properties": {
        "query": {"type": "string", "description": "message text or key phrases"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 12}},
        "required": ["query"]}
}

_model = None
def get_model():
    global _model, _fallback
    if _model is None:
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=R.MODEL)
        except Exception:
            _model = False
    return _model

def handle(msg):
    if msg.get("method") == "initialize":
        return {"protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "psychology-rag", "version": "1.0.0"}}
    if msg.get("method") == "tools/list":
        return {"tools": [TOOL]}
    if msg.get("method") == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        if name != TOOL["name"]:
            return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
        qvec = None
        m = get_model()
        if m is not False:
            qvec = list(m.embed([args["query"]]))[0]
        res = R.search(args["query"], top_k=int(args.get("top_k", R.TOP_K)), qvec=qvec)
        if not res:
            return {"content": [{"type": "text", "text": "ничего не найдено (ни lexical, ни semantic) — не выдумывать метки"}]}
        lines = [f"[{r['kind']} · {r['label']} · {r['score']}] {r['snippet']}" for r in res]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    return None

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("method") == "notifications/initialized":
            continue
        result = handle(msg)
        if result is None:
            continue
        out = {"jsonrpc": "2.0", "id": msg.get("id"), "result": result}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
