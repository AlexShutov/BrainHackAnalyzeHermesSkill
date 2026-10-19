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
  {"op":"neural","texts":[...],"models":["emotion","toxicity","sarcasm"],"batch":32}
                                                       -> {"ok":true,"scores":{name:[...]}}

PSYCH_WORKER_NO_MODELS=1 — не грузить модели: поиск только лексический, классификатор
пустой. Это режим тестов и деградации, когда ONNX/fastembed недоступны.

Жизненный цикл (замер на этой машине, 2026-10-19: worker с тремя HF-моделями — 2.34 ГБ
RSS, с fastembed и ONNX-классификатором — 3.39 ГБ на живой процесс): worker уходит по
простою (`PSYCH_WORKER_IDLE`, default 300 с), по EOF клиента, и по смерти родителя; при
превышении `PSYCH_WORKER_RSS_LIMIT_KB` (default 3 600 000 КБ — выше штатного RSS, иначе
guard срабатывал бы на первом запросе и нейро-слой никогда бы не работал) worker отдаёт
HF-модели и продолжает работать лексиконом и классификатором; после простоя нейро
(`PSYCH_WORKER_NEURAL_IDLE`, default 120 с) модели тоже освобождаются, но следующий
запрос вправе загрузить их заново. Заметка об освобождении присоединяется к ответу на
запрос, а не пишется отдельной строкой: отдельная строка сдвинула бы ответы клиента.
Клиент регистрирует atexit: выход процесса-хозяина забирает worker с собой — висячий
тёплый процесс на 16 ГБ машине это память, которую никто не отдаст до конца жизни процесса.
"""
import gc, json, os, select, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers, rag_core as R
from util import OPERATING_POINT   # рабочая точка — одна для всех путей

NO_MODELS = os.environ.get("PSYCH_WORKER_NO_MODELS") == "1"
CHUNK = R.CHUNK

# Жизненный цикл worker. Замер на этой машине (2026-10-19): живой worker с тремя
# HF-моделями — 2.34 ГБ RSS. На 16 ГБ машине это не «тёплый кэш», а риск свопa,
# поэтому worker обязан (1) уходить по простою, (2) уходить по EOF клиента,
# (3) уходить, если родитель умер, (4) отдавать модели при превышении лимита.
IDLE_TIMEOUT = int(os.environ.get("PSYCH_WORKER_IDLE", "300"))
# Лимит обязан быть ВЫШЕ штатного RSS загруженного worker. Замеры на этой машине:
# 2.34 ГБ (три HF-модели) и 3.39 ГБ (fastembed + ONNX-классификатор + три HF-модели в
# одном процессе). Иначе guard срабатывает на первом же запросе и нейро-слой никогда не
# работает — это и произошло при живой проверке: анализ загрузил модели, guard тут же их
# отдал, и доставляемые числа получил словарь. Guard — сигнал утечки или роста, а не
# выключение слоя; «не сидеть в памяти» обеспечивает выход по простою (IDLE_TIMEOUT).
RSS_LIMIT_KB = int(os.environ.get("PSYCH_WORKER_RSS_LIMIT_KB", "3600000"))
# Память на паузе: долгий хозяин (гейтвей) живёт часами, и 3.4 ГБ модельного RSS в нём —
# занятая память, а не тёплый кэш. Модели освобождаются после простоя, процесс остаётся
# (лексикон + классификатор), следующий запрос вправе загрузить заново.
NEURAL_IDLE_RELEASE = int(os.environ.get("PSYCH_WORKER_NEURAL_IDLE", "120"))


def current_rss_kb():
    """Текущий RSS, а не пик. `ru_maxrss` на macOS — максимум за жизнь процесса, он
    никогда не уменьшается: пиковый guard сработал бы один раз и остался «вечно
    превышенным» даже после освобождения моделей."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def attach_pending(resp, pending):
    """Заметка об освобождении памяти едет В ОТВЕТЕ на запрос, а не отдельной строкой:
    строка, посланная между запросами, сдвинула бы ответы клиента (тот же баг, что и с
    отдельным emit guard)."""
    if not pending:
        return False
    resp["memory"] = pending
    return True


def idle_release_due(last_neural, now, seconds=None):
    """Пора освободить модели после простоя (процесс остаётся). Отдельно от guard:
    guard — превышение лимита и постоянный отказ от перезагрузки, здесь — память на
    паузе, и следующий запрос вправе загрузить модели заново."""
    return (now - last_neural) >= (NEURAL_IDLE_RELEASE if seconds is None else seconds)


def guard_memory(resp, state):
    """Память добавляется к ТОМУ ЖЕ ответу, а не отдельным сообщением. Отдельный emit
    сдвигает ответы клиента: следующий запрос читает строку предыдущего (замер живым
    прогоном — после guard ответ на `markers` пришёл как `neural released`). Возвращает
    True, если освобождение произошло."""
    if not state or not should_release(current_rss_kb(), RSS_LIMIT_KB):
        return False
    rss = current_rss_kb()
    freed = release_neural(state)
    resp["memory"] = {"rss_kb": rss, "released_kb": freed, "limit_kb": RSS_LIMIT_KB,
                      "neural_released": True}
    return True


def neural_gate(neural, released):
    """Что делать на запросе neural. `released` важнее пустого состояния: память уже
    отдана по guard, и молча перезагрузить модели нельзя — это вернуло бы те же гигабайты,
    ради которых их освободили. Без этого порядка пустой dict после освобождения
    интерпретировался бы как «загрузить заново»."""
    if released:
        return "released"
    if not neural:
        return "load"
    return "run"


def parent_changed(ppid0):
    """Родитель умер — сирота переродится (getppid изменится на launchd/init). Без этой
    проверки worker переживает хозяина и держит модели после его смерти."""
    return os.getppid() != ppid0


def should_release(rss_kb, limit_kb):
    return rss_kb > limit_kb


def release_neural(state):
    """Освободить модели и вернуть количество освобождённых КБ (оценка по RSS
    до и после gc). Возвращает 0, если освобождать нечего."""
    if not state:
        return 0
    before = current_rss_kb()
    state.clear()
    gc.collect()
    return max(0, before - current_rss_kb())


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


NEURAL_MODELS = ("emotion", "toxicity", "sarcasm")


def neural_paths():
    """Дом neural_pass — youtube-content/scripts. Отдельная функция, потому что путь
    обязан быть проверяемым: без него импорт падает, worker отвечает «neural
    unavailable», и путь комментариев молча возвращается к загрузке моделей в каждом
    процессе — ровно тот баг, который тест ловит."""
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(skill_dir, "youtube-content", "scripts")


def load_neural():
    """HF-модели нейро-прохода комментариев — загружаются ОДИН раз на живой процесс
    worker и лениво, на первом запросе `neural`: иначе старт worker платит загрузку
    rubert для тех, кто нейро-слой не просит. Правило навыка: warm worker — один на
    хозяина, и через него обязаны ходить ВСЕ пути, а не только плагин."""
    if NO_MODELS:
        return None
    try:
        sys.path.insert(0, neural_paths())
        import neural_pass as NP
        from transformers import AutoTokenizer
        loaded = {}
        for name in NEURAL_MODELS:
            tok = AutoTokenizer.from_pretrained(NP.REPOS[name])
            model, labels = NP.load_no_mmap(NP.REPOS[name])
            loaded[name] = (tok, model, labels, NP.MULTILABEL[name])
        return loaded
    except Exception:
        return None


def main():
    t0 = time.time()
    con = R.open_db()
    emb = load_embedding()
    cls = load_classifier()
    matrix = None
    neural = None
    neural_released = False

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

    def dispatch(req):
        """Один ответ на один запрос. Ops обязаны ВОЗВРАЩАТЬ dict, а не emit: иначе
        memory guard пишет вторую строку и клиент читает ответ не своего запроса."""
        nonlocal neural, neural_released
        op = req.get("op")
        if op == "ping":
            return {"ok": True}
        if op == "search":
            q = req.get("query") or ""
            chunks = req.get("chunks") or [q]
            qvecs = [list(emb.embed([c]))[0] for c in chunks] if emb else [None] * len(chunks)
            return {"ok": True, "results": R.merge_chunked(chunks, con=con, qvecs=qvecs,
                                                           matrix=cached_matrix())}
        if op == "markers":
            return {"ok": True, "markers": markers.scan(req.get("text") or "")}
        if op == "classify":
            text = req.get("text") or ""
            minp = float(req.get("min", OPERATING_POINT))
            if cls is None:
                return {"ok": True, "classification": [], "model": None}
            if req.get("gated"):
                res = []
                for i, p in cls.gated_paragraphs(text):
                    labels = cls.classify(p, min_conf=minp)
                    if labels:
                        res.append({"para": i, "text": p, "labels": labels})
                return {"ok": True, "classification": res}
            return {"ok": True, "classification": cls.classify(text, min_conf=minp)}
        if op == "neural":
            if NO_MODELS:
                return {"ok": False, "error": "no models"}
            state = neural_gate(neural, neural_released)
            if state == "released":
                return {"ok": False, "error": "neural released by memory guard",
                        "released": True}
            if state == "load":
                neural = load_neural()
            if not neural:
                return {"ok": False, "error": "neural unavailable"}
            import neural_pass as NP
            texts = req.get("texts") or []
            models = req.get("models") or list(NEURAL_MODELS)
            batch = int(req.get("batch") or NP.BATCH)
            out = {}
            for name in models:
                if name not in neural:
                    continue
                tok, model, labels, multi = neural[name]
                out[name] = [row["all"] for row in
                             NP.infer(tok, model, labels, texts, multi, name, batch)]
            return {"ok": True, "scores": out, "loaded_once": True}
        return {"ok": False, "error": f"unknown op: {op}"}

    emit({"ok": True, "warm_ms": round((time.time() - t0) * 1000),
          "embedding": emb is not None, "classifier": cls is not None,
          "docs": con.execute("SELECT COUNT(*) FROM docs").fetchone()[0],
          "idle_s": IDLE_TIMEOUT, "rss_limit_kb": RSS_LIMIT_KB})

    ppid0 = os.getppid()
    stdin_fd = sys.stdin.fileno()
    buf = b""
    last_request = time.time()
    last_neural = last_request
    pending = None
    while True:
        if b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
        else:
            r, _, _ = select.select([stdin_fd], [], [], min(IDLE_TIMEOUT, NEURAL_IDLE_RELEASE))
            if not r:
                now = time.time()
                if os.environ.get("PSYCH_WORKER_DEBUG"):
                    sys.stderr.write(f"[idle] now-last_request={now-last_request:.1f} "
                                     f"idle={IDLE_TIMEOUT} neural_idle={NEURAL_IDLE_RELEASE} "
                                     f"neural={'yes' if neural else 'no'}\n")
                if now - last_request >= IDLE_TIMEOUT:
                    emit({"ok": False, "error": "idle timeout", "idle_s": IDLE_TIMEOUT})
                    break
                if neural and idle_release_due(last_neural, now):
                    # память на паузе: долгий хозяин (гейтвей) живёт часами, и 3.4 ГБ
                    # модельного RSS в нём — это не «тёплый кэш», а занятая память.
                    # Освобождаем модели, процесс остаётся (лексикон + классификатор),
                    # и следующий запрос вправе загрузить заново — это не guard
                    freed = release_neural(neural)
                    pending = {"released_kb": freed, "idle_release": True}
                continue
            chunk = os.read(stdin_fd, 65536)
            if not chunk:
                # EOF: клиент умер. Worker обязан уйти, а не сидеть с моделями —
                # на 16 ГБ машине висячий тёплый процесс это гигабайты, которые никто
                # не отдаст до конца жизни процесса
                break
            buf += chunk
            continue
        if not line.strip():
            continue
        if os.getppid() != ppid0:
            emit({"ok": False, "error": "parent gone"})
            break
        try:
            req = json.loads(line)
        except Exception:
            emit({"ok": False, "error": "bad json"})
            continue
        last_request = time.time()
        if req.get("op") == "neural":
            last_neural = last_request
        resp = dispatch(req)
        if attach_pending(resp, pending):
            pending = None
        if guard_memory(resp, neural):
            neural_released = True
        emit(resp)
    con.close()


if __name__ == "__main__":
    main()
