"""Плагин: /analyze <ссылка на видео>.

Контракт обработчика плагин-команд: ``fn(raw_args) -> str | None``; гейтвей доставляет
возвращённую строку как ответ, ``None`` — ничего не доставляет. Обработчик возвращает None,
потому что сообщение доставляет скрипт одним сообщением — второй копии быть не должно.

Чат берётся из сессионного контекста гейтвея, чтобы команда работала в том чате, где её
набрали; вне гейтвея — ANALYZE_CHAT_ID, иначе значение по умолчанию.

Плагин не тянет модель: он запускает скрипт. Нейросети используются только на своих шагах
пайплайна (классификатор, нейро-слой комментариев).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
SKILL_NAME = "brain-hack-analyze"
DEFAULT_CHAT = "362122320"

SPAWN = None   # подменяемая точка вызова subprocess: без mock проверка сводится к чтению исходника


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


def resolve_chat():
    try:
        from gateway.session_context import get_session_env
        chat = get_session_env("HERMES_SESSION_CHAT_ID") or ""
    except Exception:
        chat = os.environ.get("HERMES_SESSION_CHAT_ID") or ""
    return chat or os.environ.get("ANALYZE_CHAT_ID") or DEFAULT_CHAT


def _default_spawn(env, url):
    return subprocess.Popen([sys.executable, SCRIPT, url, "--run-queue"], env=env)


def run(raw_args="", chat=None, spawn=None):
    url = (raw_args or "").strip()
    if not url:
        return "нужна ссылка: /analyze <ссылка на видео>"
    chat = chat or resolve_chat()
    env = dict(os.environ)
    env["ANALYZE_CHAT_ID"] = chat
    if not os.path.exists(SCRIPT):
        # путь обязан называться: без этого пользователь видит «не запустилось» и не
        # понимает, что скрипт анализа вообще не найден
        return "не найден скрипт анализа: " + SCRIPT
    try:
        proc = spawn(env, url) if spawn else (SPAWN or _default_spawn)(env, url)
    except Exception as exc:
        return "не запустилось: %s" % exc
    # returncode читается ПОСЛЕ завершения процесса. Прежний код читал его у незавершённого
    # Popen: там он None, None != 0 — и «ОШИБКА» сообщалась на каждом вызове, даже когда
    # скрипт работал нормально.
    rc = proc.wait() if hasattr(proc, "wait") else getattr(proc, "returncode", 0)
    if rc != 0:
        err = (getattr(proc, "stderr", "") or "").strip()
        return "ОШИБКА: " + (err or "скрипт завершился с кодом %s" % rc)
    # сообщение доставляет скрипт одним сообщением; обработчик возвращает None
    return None


def register(ctx):
    ctx.register_command(
        "analyze",
        lambda args: run(args),
        description="Анализ видео по ссылке: очередь, блоки по мере доступности, "
                    "доставка скриптом без хода агента.",
    )
