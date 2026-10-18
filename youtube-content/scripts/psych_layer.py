#!/usr/bin/env python3
"""Psychology layer of the video pipeline: run bigbro-analyze on the transcript.

Calls the three layers of the sibling bigbro-analyze skill (lexical markers →
RAG retrieval → ONNX classifier) on the raw transcript and returns one combined
JSON. Transcript-only: this is how bigbro-analyze participates in video
analysis — never applied to the chat message itself.

Exit 0 always; on any problem returns {"status": "unavailable", "reason": ...}
so the video pipeline degrades to the plain ⚫ passes without breaking.

Usage:
  echo "$TRANSCRIPT" | python3 psych_layer.py
  python3 psych_layer.py --file transcript.txt
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIGBRO = os.path.abspath(os.path.join(HERE, "..", ".."))
HERMES = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
PY = os.environ.get("PSYCH_RAG_PY", os.path.join(HERMES, "tools", "psych-rag-env", "bin", "python"))
SCRIPTS = os.path.join(BIGBRO, "scripts")


def call(script, args, timeout):
    try:
        out = subprocess.run([PY, os.path.join(SCRIPTS, script)] + args,
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="transcript file (default: stdin)")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    text = (text or "").strip()
    if not text:
        print(json.dumps({"status": "empty", "reason": "пустой транскрипт"}, ensure_ascii=False))
        return

    if not (os.path.isfile(PY) and os.path.isfile(os.path.join(SCRIPTS, "markers.py"))):
        print(json.dumps({"status": "unavailable",
                          "reason": f"bigbro scripts or venv not found ({SCRIPTS})"},
                         ensure_ascii=False))
        return

    markers = call("markers.py", [text, "--json"], 30)
    classifier = call("classifier.py", [text, "--json"], 120)
    rag = call("rag_query.py", [text[:1500], "--top", "6"], 60)

    result: dict = {"status": "ok", "source": "transcript"}
    if markers:
        result["markers_hits"] = markers.get("hits", 0)
        result["markers"] = markers.get("markers", [])
    if classifier and classifier.get("classification"):
        result["classification"] = classifier["classification"]
    if rag and rag.get("results"):
        result["rag"] = [{"kind": r["kind"], "label": r["label"], "score": r["score"],
                          "snippet": r["snippet"]} for r in rag["results"]]
    if "classification" not in result and "markers" not in result:
        result["status"] = "unavailable"
        result["reason"] = "слои bigbro не ответили"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
