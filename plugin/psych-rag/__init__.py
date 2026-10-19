"""psych-rag plugin: guaranteed RAG injection for the brain-hack-analyze skill.

Uses the supported `pre_gateway_dispatch` hook (policy hook, supports
{"action": "rewrite", "text"}) — no core patch. When an incoming message is
multi-paragraph (the skill's auto-trigger), retrieval runs against the local
psychology index and the top-k block is appended to the message text before
dispatch, so even a weak local model that forgets to call the MCP tool still
receives the passages.

Retrieval is executed as a subprocess against the RAG venv (fastembed/ONNX), so
the gateway environment stays clean; BM25-only fallback if ONNX cannot load.
All three layers receive the FULL message text up to the gate limit (4096 chars — the
platform limit; it used to be 4000, so messages 4001-4096 were never analyzed at all) —
earlier they got only the first 1500, so messages between 1500 and the gate passed the
gate but were analyzed on a fragment (signals after 1500 were invisible).
"""
import json, os, re, subprocess, sys, time

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
MAX_PARAS = 12          # реальный гейт: не переписывать сообщения длиннее 12 абзацев
MIN_PARAS = 2           # the skill's auto-trigger threshold
MAX_CHARS = 4096        # гейт обязан равняться фактическому лимиту платформы: при 4000
                        # сообщения 4001–4096 символов не разбирались вообще, и это
                        # выглядело как пропуск анализа, а не как отказ гейта
TIMEOUT = 12
# рабочая точка — одна для всех путей (util); раньше каждый модуль держал свою
RECHECK_SECONDS = 86400  # перепроверка версии модели не чаще раза в сутки на чат
_checked = {}            # (platform, chat_id) -> время проверки

sys.path.insert(0, SKILL_DIR + "/scripts")
try:
    from util import OPERATING_POINT   # рабочая точка правила — одна для всех путей
except Exception:
    OPERATING_POINT = 0.70
try:
    import markers as _markers   # каноническое определение абзаца — одна для всех слоёв
except Exception:
    _markers = None

try:
    import worker_client as _wc   # тёплый worker: модели загружены один раз
except Exception:
    _wc = None


def worker_call(op, **kw):
    """Один живой процесс вместо subprocess на каждое сообщение.
    Замер: холодный путь ~2.7 с на сообщение (1.54 с embedding + 1.17 с ONNX + 30 мс маркеры),
    тёплый ~50-70 мс. Если worker не поднимается — возврат None, вызывающий идёт
    subprocess-путём (деградация, не отказ)."""
    if _wc is None:
        return None
    try:
        return _wc.client(RAG_PY).call(op, **kw)
    except Exception:
        return None


def chunks_of(text, size=1500):
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


def paragraphs(text):
    if _markers is not None:
        return _markers.paragraphs(text)
    return [p for p in re.split(r"\n+", text) if p.strip()]


def retrieve(text):
    r = worker_call("search", query=text, chunks=chunks_of(text))
    if r and r.get("results"):
        return r["results"]
    try:
        # таймаут был рассчитан на срез 1500; на полном тексте embedding занимает
        # ~6 с (замер: 3.2 с на 1500, 6.2 с на 4000), поэтому запас увеличен
        out = subprocess.run([RAG_PY, QUERY, text, "--chunk", "1500"], capture_output=True, text=True,
                             timeout=TIMEOUT * 2)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        return data.get("results") or None
    except Exception:
        return None


def marker_hits(text):
    """Surface signals present in the text itself — no terminology required."""
    r = worker_call("markers", text=text)
    if r and r.get("markers"):
        return r["markers"]
    try:
        out = subprocess.run([RAG_PY, MARKERS, text, "--json"],
                             capture_output=True, text=True, timeout=TIMEOUT)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        return data.get("markers") or None
    except Exception:
        return None


def version_check(source):
    """Перепроверка версии модели: один раз на чат, не чаще раза в сутки.

    Границы сессии плагину недоступны (нет доступа к session key), поэтому ключ —
    (platform, chat_id): новый чат получает свою проверку, прежний пересматривается
    не раньше, чем через сутки.
    """
    key = (getattr(source, "platform", None), getattr(source, "chat_id", None))
    now = time.time()
    last = _checked.get(key)
    if last is not None and now - last < RECHECK_SECONDS:
        return None
    _checked[key] = now
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
    r = worker_call("classify", text=text, min=OPERATING_POINT, gated=True)
    if r and r.get("classification"):
        return r["classification"]
    try:
        out = subprocess.run([RAG_PY, CLASSIFIER, text, "--json", "--gated",
                              "--min", str(OPERATING_POINT)],
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
    # Guard по всем четырём маркерам, не только по первому: пересланное сообщение,
    # которое уже содержит блок маркеров/классификатора, иначе переписывается повторно
    # и блок появляется дважды (или начинается рекурсия: rewrite -> новый inbound).
    if getattr(event, "internal", False) or any(m in text for m in (MARK, MARK2, MARK3, MARK4)):
        return None
    paras = paragraphs(text)
    if len(paras) < MIN_PARAS or len(paras) > MAX_PARAS or len(text) > MAX_CHARS:
        return None
    results = retrieve(text)
    hits = marker_hits(text)
    notice = version_check(getattr(event, "source", None))
    cls = classifier_hits(text, hits)
    if not results and not hits and not cls and not notice:
        return None
    blocks = []
    if results:
        lines = [f"[{r['kind']} · {r['label']} · {r['score']}] {r['snippet']}" for r in results]
        blocks.append(MARK + str(len(lines)) + "]\n" + "\n".join(lines))
    if hits:
        lines = [f"{h['family']} / {h['marker']}: [{h['found']}] {h['context']}" for h in hits[:12]]
        blocks.append(MARK2 + "\n" + "\n".join(lines))
    if cls:
        lines = [f"абзац {r['para']}: " + ", ".join(f"{l['ru']} ({l['p']})" for l in r["labels"])
                 for r in cls]
        blocks.append(MARK3 + "\n" + "\n".join(lines))
    if notice:
        blocks.append(MARK4 + "\n" + notice.get("announcement", ""))
    # Итог не должен превышать лимит платформы: исходный текст сохраняется целиком,
    # усекаются блоки (они — служебное добавление, а не содержание сообщения)
    new_text = text
    for b in blocks:
        if len(new_text) + len(b) > MAX_CHARS:
            room = MAX_CHARS - len(new_text)
            if room > 0:
                new_text += b[:room]
            break
        new_text += b
    return {"action": "rewrite", "text": new_text}


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", callback)
