#!/usr/bin/env python3
"""Клиент тёплого worker (`scripts/worker.py`) — для плагина и psych_layer.

Один живой процесс на хозяина: worker держит fastembed и ONNX загруженными, поэтому
сообщение перестаёт платить загрузку модели каждый раз (замер: холодный путь ~2.7 с на
сообщение, тёплый ~50-70 мс).

При смерти процесса — один перезапуск. Если worker не поднимается, вызывающий падает
обратно на subprocess-путь: деградация, а не отказ анализа.

Чтение — бинарное, через select: у TextIOWrapper собственный буфер, и select на fd при
буферизованной строке показал бы «нет данных», а readline завис бы. Здесь буфер свой, и
select работает честно.
"""
import json, os, select, subprocess, sys

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(SKILL_DIR, "scripts", "worker.py")
DEFAULT_PY = os.environ.get("PSYCH_RAG_PY") or sys.executable
TIMEOUT = int(os.environ.get("PSYCH_WORKER_TIMEOUT", "15"))


class WorkerClient:
    def __init__(self, python=None, timeout=None):
        self.python = python or DEFAULT_PY
        self.timeout = timeout or TIMEOUT
        self.p = None
        self.buf = b""
        self.hello = None

    def start(self):
        try:
            self.p = subprocess.Popen([self.python, WORKER],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      bufsize=0)
            self.buf = b""
            self.hello = self._read()
            return self.hello
        except Exception:
            self.stop()
            return None

    def _read(self):
        if self.p is None:
            return None
        while b"\n" not in self.buf:
            r, _, _ = select.select([self.p.stdout], [], [], self.timeout)
            if not r:
                return None
            chunk = os.read(self.p.stdout.fileno(), 65536)
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            return json.loads(line.decode("utf-8"))
        except Exception:
            return None

    def call(self, op, **kw):
        if self.p is None and self.start() is None:
            return None
        req = json.dumps({"op": op, **kw}, ensure_ascii=False).encode("utf-8")
        for attempt in (1, 2):
            try:
                self.p.stdin.write(req + b"\n")
                self.p.stdin.flush()
                r = self._read()
                if r is not None:
                    return r
            except Exception:
                pass
            if attempt == 1:
                self.stop()
                if self.start() is None:
                    return None
            else:
                self.stop()
        return None

    def stop(self):
        if self.p is not None:
            for stream in (self.p.stdin, self.p.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                self.p.wait(timeout=2)
            except Exception:
                try:
                    self.p.kill()
                except Exception:
                    pass
            self.p = None
        self.buf = b""
        self.hello = None


_client = None


def client(python=None):
    """Общий клиент на процесс-хозяина; интерпретатор фиксированный (venv с fastembed)."""
    global _client
    if _client is None or (python and _client.python != python):
        _client = WorkerClient(python)
    return _client
