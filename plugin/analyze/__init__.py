"""Плагин: /analyze <ссылка на видео>.

Контракт обработчика плагин-команд: ``fn(raw_args) -> str | None``; гейтвей доставляет
возвращённую строку как ответ, ``None`` — ничего не доставляет.

Два пути под одним интерфейсом, и это главное правило навыка:

1. Если Python-скрипт анализа есть (версия навыка с конвейером) — запускается он, сообщение
   доставляет скрипт одним сообщением, обработчик возвращает None (второй копии быть не должно).
2. Если скрипта нет (версия навыка без конвейера) — ссылка обязана попасть в ход агента через
   inject_message. Dispatch плагин-команды идёт ДО хода агента, и None означает «ничего не
   доставлено»: ссылка никуда не попадает, и навык её не видит. Поэтому обработчик не может
   просто вернуть None — он обязан передать ссылку сессии.

Чат берётся из сессионного контекста гейтвея, чтобы команда работала в том чате, где её
набрали; вне гейтвея — ANALYZE_CHAT_ID, иначе значение по умолчанию.

Плагин не тянет модель: он либо запускает скрипт, либо передаёт ссылку сессии. Нейросети
используются только на своих шагах пайплайна.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
SKILL_NAME = "brain-hack-analyze"
DEFAULT_CHAT = "362122320"

SPAWN = None    # подменяемая точка вызова subprocess
INJECT = None   # подменяемая точка передачи ссылки сессии

URL_RE = re.compile(r"https?://\S+")
VIDEO_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{11}\Z")


def script_candidates():
    """Плагин загружается из ~/.hermes/plugins/<name> — это отдельная копия, а не симлинк
    навыка. Прежний путь решался от расположения файла: для живой копии это
    ~/.hermes/scripts/analyze.py, которого нет, и /analyze падала на запуске скрипта.
    Порядок: runtime-копия (её исполняет установка), затем копия навыка, затем путь от
    файла — если плагин загружен прямо из навыка.
    """
    return [os.path.join(HERMES_HOME, "scripts", "analyze.py"),
            os.path.join(HERMES_HOME, "skills", SKILL_NAME, "scripts", "analyze.py"),
            os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts", "analyze.py")]


def resolve_script():
    for c in script_candidates():
        if os.path.exists(c):
            return c
    return script_candidates()[0]


SCRIPT = resolve_script()


def extract_url(raw):
    """Ссылка извлекается из аргумента, а не читается как путь целиком: аргумент может быть
    «Анализ https://...» или «https://... с таймкодами». Без извлечения ссылка теряется."""
    text = (raw or "").strip()
    m = URL_RE.search(text)
    if m:
        return m.group(0).rstrip(").,;")
    if VIDEO_ID_RE.match(text):
        return "https://www.youtube.com/watch?v=" + text
    return ""


def session_key():
    try:
        from gateway.session_context import get_session_env
        key = get_session_env("HERMES_SESSION_KEY") or ""
    except Exception:
        key = os.environ.get("HERMES_SESSION_KEY") or ""
    return key


def resolve_chat():
    try:
        from gateway.session_context import get_session_env
        chat = get_session_env("HERMES_SESSION_CHAT_ID") or ""
    except Exception:
        chat = os.environ.get("HERMES_SESSION_CHAT_ID") or ""
    return chat or os.environ.get("ANALYZE_CHAT_ID") or DEFAULT_CHAT


def _default_spawn(env, url):
    return subprocess.Popen([sys.executable, SCRIPT, url, "--run-queue"], env=env)


def run(raw_args="", ctx=None, chat=None, spawn=None, inject=None, exists=None):
    url = extract_url(raw_args)
    if not url:
        return "нужна ссылка: /analyze <ссылка на видео>"

    exists = exists or os.path.exists
    if exists(SCRIPT):
        chat = chat or resolve_chat()
        env = dict(os.environ)
        env["ANALYZE_CHAT_ID"] = chat
        try:
            proc = spawn(env, url) if spawn else (SPAWN or _default_spawn)(env, url)
        except Exception as exc:
            return "не запустилось: %s" % exc
        # returncode читается ПОСЛЕ завершения процесса: у незавершённого Popen его нет,
        # и прежний код сообщал «ОШИБКА» на каждом вызове, даже когда скрипт работал
        rc = proc.wait() if hasattr(proc, "wait") else getattr(proc, "returncode", 0)
        if rc != 0:
            err = (getattr(proc, "stderr", "") or "").strip()
            return "ОШИБКА: " + (err or "скрипт завершился с кодом %s" % rc)
        # сообщение доставляет скрипт одним сообщением; обработчик возвращает None
        return None

    # Конвейера нет — навык исполняется ходом агента. Ссылка обязана дойти до него.
    inject = inject or INJECT or (getattr(ctx, "inject_message", None) if ctx else None)
    if inject is None:
        return "конвейера анализа нет и ход агента недоступен: напиши «Анализ %s»" % url
    key = session_key()
    if not key:
        return "не могу передать ссылку навыку: нет ключа сессии — напиши «Анализ %s»" % url
    try:
        ok = inject("Анализ " + url, "user", session_key=key)
    except Exception as exc:
        return "не могу передать ссылку навыку: %s — напиши «Анализ %s»" % (exc, url)
    if not ok:
        return ("навык не получил ссылку: inject запрещён для плагина или нет живого гейтвея "
                "(нужно plugins.entries.analyze.allow_gateway_injection: true) — напиши «Анализ %s»" % url)
    return None


def register(ctx):
    ctx.register_command(
        "analyze",
        lambda args: run(args, ctx),
        description="Анализ видео по ссылке: конвейер, если он есть; иначе ссылка передаётся "
                    "ходу агента, чтобы навык её увидел.",
    )
