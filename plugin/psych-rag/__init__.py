"""psych-rag plugin: guaranteed RAG injection for the brain-hack-analyze skill.

Uses the supported `pre_gateway_dispatch` hook (policy hook, supports
{"action": "rewrite", "text"}) — no core patch. When an incoming message is
multi-paragraph (the skill's auto-trigger), retrieval runs against the local
psychology index and the top-k block is appended to the message text before
dispatch, so even a weak local model that forgets to call the MCP tool still
receives the passages.

Retrieval is executed as a subprocess against the RAG venv (fastembed/ONNX), so
the gateway environment stays clean; BM25-only fallback if ONNX cannot load.
All three layers receive the FULL message text up to the gate limit (4000 chars) —
earlier they got only the first 1500, so messages between 1500 and 4000 passed the
gate but were analyzed on a fragment (signals after 1500 were invisible).
"""
import json, os, subprocess, time

HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
SKILL = "brain-hack-analyze"
SKILL_DIR = os.environ.get("PSYCH_SKILL_DIR") or os.path.join(HOME, "skills", SKILL)
RAG_PY = os.environ.get("PSYCH_RAG_PY", os.path.join(HOME, "tools", "psych-rag-env", "bin", "python"))
QUERY = SKILL_DIR + "/scripts/rag_query.py"
MARKERS = SKILL_DIR + "/scripts/markers.py"
CLASSIFIER = SKILL_DIR + "/scripts/classifier.py"
CHECK = SKILL_DIR + "/scripts/model_update_check.py"
MARK = "\n\n[RAG · психология · топ-"
MARK2 = "\n\n[маркеры в тексте]"
MARK3 = "\n\n[классификатор (второе мнение)]"
MARK4 = "\n\n[обновление модели]"
MAX_PARAS = 12          # do not rewrite very long messages
MIN_PARAS = 2           # the skill's auto-trigger threshold
TIMEOUT = 12
_checked = False


def paragraphs(text):
    return [p for p in text.split("\n") if p.strip()]


def retrieve(text):
    try:
        # таймаут был рассчитан на срез 1500; на полном тексте embedding занимает
        # ~6 с (замер: 3.2 с на 1500, 6.2 с на 4000), поэтому запас увеличен
        out = subprocess.run([RAG_PY, QUERY, text], capture_output=True, text=True,
                             timeout=TIMEOUT * 2)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        return data.get("results") or None
    except Exception:
        return None


def marker_hits(text):
    """Surface signals present in the text itself — no terminology required."""
    try:
        out = subprocess.run([RAG_PY, MARKERS, text, "--json"],
                             capture_output=True, text=True, timeout=TIMEOUT)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        return data.get("markers") or None
    except Exception:
        return None


def version_check():
    """Once per session: compare the stored model version with the hub.

    If an update is warranted the worker marks it as needed and downloads in the
    background; the model in use keeps serving until the new one is verified.
    """
    global _checked
    if _checked:
        return None
    _checked = True
    try:
        args = [RAG_PY, CHECK, "--json"]
        if os.environ.get("PSYCH_RAG_DRY_RUN") == "1":
            args.append("--dry-run")
        out = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT * 2)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        if data.get("status") in ("needed", "updating", "ready", "downloaded_unverified"):
            return data
    except Exception:
        return None
    return None


def classifier_hits(text, hits):
    """Second opinion: fine-tuned classifier, gated by the marker layer.

    Runs only on paragraphs where markers fired — no hits, no model load.
    Measured: 19 ms per paragraph warm, 0.5 s session load, peak ~1.3 GB.
    On by default; disable with PSYCH_RAG_CLASSIFIER=0.
    """
    if os.environ.get("PSYCH_RAG_CLASSIFIER", "1") != "1" or not hits:
        return None
    try:
        out = subprocess.run([RAG_PY, CLASSIFIER, text, "--json", "--gated"],
                             capture_output=True, text=True, timeout=TIMEOUT * 3)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        return data.get("classification") or None
    except Exception:
        return None


def callback(*args, **kwargs):
    event = kwargs.get("event")
    if event is None:
        return None
    text = getattr(event, "text", "") or ""
    if getattr(event, "internal", False) or MARK in text:
        return None
    if len(paragraphs(text)) < MIN_PARAS or len(text) > 4000:
        return None
    results = retrieve(text)
    hits = marker_hits(text)
    notice = version_check()
    cls = classifier_hits(text, hits)
    if not results and not hits and not cls and not notice:
        return None
    new_text = text
    if results:
        lines = [f"[{r['kind']} · {r['label']} · {r['score']}] {r['snippet']}" for r in results]
        new_text += MARK + str(len(lines)) + "]\n" + "\n".join(lines)
    if hits:
        lines = [f"{h['family']} / {h['marker']}: [{h['found']}] {h['context']}" for h in hits[:12]]
        new_text += MARK2 + "\n" + "\n".join(lines)
    if cls:
        lines = [f"абзац {r['para']}: " + ", ".join(f"{l['ru']} ({l['p']})" for l in r["labels"])
                 for r in cls]
        new_text += MARK3 + "\n" + "\n".join(lines)
    if notice:
        new_text += MARK4 + "\n" + notice.get("announcement", "")
    return {"action": "rewrite", "text": new_text}


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", callback)
