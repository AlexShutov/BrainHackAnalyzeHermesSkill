#!/usr/bin/env python3
"""Пост-статус для долгих задач (одно сообщение, редактируется по мере работы).

Usage:
  status_post.py start <chat_id> <key> <preview> <status>
  status_post.py update <key> <status>
  status_post.py done  <key>

start: отправляет пост «<preview>\n\nСтатус:\n<status>» и сохраняет message_id.
update: правит текст статуса в том же сообщении.
done:   ставит статус «Готово».
Ключу <key> соответствует файл ~/.hermes/state/skill_status/<key>.json
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

HOME = os.path.expanduser("~/.hermes")
STATE_DIR = os.path.join(HOME, "state", "skill_status")
TG_LIMIT = 4096          # лимит sendMessage/editMessageText — превышение даёт HTTP 400


def bot_token():
    with open(os.path.join(HOME, ".env")) as f:
        for line in f:
            if line.startswith(("TELEGRAM_BOT_TOKEN=", "BOT_TOKEN=")):
                return line.split("=", 1)[1].strip()
    raise SystemExit("токен бота не найден в ~/.hermes/.env")


def api(method, **params):
    """Один запрос с обработкой 429 и сетевой ошибки. Раньше любое HTTP-исключение
    роняло скрипт стеком, а флуд-контроль Telegram (3 edits/сек) — это обычная ситуация
    при heartbeat-обновлениях статуса."""
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{bot_token()}/{method}", data=data)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code == 429:
                try:
                    wait = int(json.loads(body).get("parameters", {}).get("retry_after", 3))
                except Exception:
                    wait = 3
                time.sleep(min(wait, 30))
                continue
            sys.exit(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                sys.exit(f"сеть: {e}")
            time.sleep(2)
    sys.exit("флуд-контроль: попытки исчерпаны")


def path(key):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{key}.json")


def text_of(preview, status):
    preview = (preview or "").strip()
    return (preview + "\n\n" if preview else "") + "Статус:\n" + (status or "").strip()


def clamp(preview, status):
    """Текст поста в пределах лимита Telegram. Превью всегда сохраняется целиком (правило
    навыка: превью — верх поста), усекается статус. Без этого длинный статус даёт HTTP 400
    и пост зависает на последом успешном обновлении."""
    t = text_of(preview, status)
    if len(t) <= TG_LIMIT:
        return t
    head = text_of(preview, "")
    room = TG_LIMIT - len(head) - 1
    if room < 1:
        return t[:TG_LIMIT]
    return head + (status or "").strip()[:room - 1] + "…"


def render(st, status):
    # превью ВСЕГДА вверху, затем подзаголовок Статус
    return clamp(st.get("preview"), status)


USAGE = "usage: status_post.py start <chat_id> <key> <preview> <status> | update <key> <status> | done <key>"


def main():
    if len(sys.argv) < 2:
        sys.exit(USAGE)
    cmd = sys.argv[1]
    if cmd == "start":
        if len(sys.argv) < 6:
            sys.exit(USAGE)
        chat_id, key, preview, status = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        r = api("sendMessage", chat_id=chat_id, text=clamp(preview, status))
        if not r.get("ok"):
            sys.exit(f"send failed: {r}")
        mid = r["result"]["message_id"]
        with open(path(key), "w") as f:
            json.dump({"chat_id": chat_id, "message_id": mid, "preview": preview.strip()}, f)
        print(f"started: {key} -> message {mid}")
    elif cmd in ("update", "done"):
        if len(sys.argv) < 3:
            sys.exit(USAGE)
        key = sys.argv[2]
        status = sys.argv[3] if cmd == "update" else "Готово"
        p = path(key)
        if not os.path.exists(p):
            sys.exit(f"нет поста для ключа {key} — сначала start")
        with open(p) as f:
            st = json.load(f)
        r = api("editMessageText", chat_id=st["chat_id"], message_id=st["message_id"],
                text=render(st, status))
        if not r.get("ok") and "message is not modified" not in r.get("description", ""):
            sys.exit(f"edit failed: {r}")
        print(f"updated: {key}")
    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
