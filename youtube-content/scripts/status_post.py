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
import json, os, sys, urllib.request, urllib.parse

HOME = os.path.expanduser("~/.hermes")
STATE_DIR = os.path.join(HOME, "state", "skill_status")


def bot_token():
    with open(os.path.join(HOME, ".env")) as f:
        for line in f:
            if line.startswith(("TELEGRAM_BOT_TOKEN=", "BOT_TOKEN=")):
                return line.split("=", 1)[1].strip()
    raise SystemExit("токен бота не найден в ~/.hermes/.env")


def api(method, **params):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{bot_token()}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def path(key):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{key}.json")


def text_of(preview, status):
    preview = (preview or "").strip()
    return (preview + "\n\n" if preview else "") + "Статус:\n" + (status or "").strip()


def render(st, status):
    # превью ВСЕГДА вверху, затем подзаголовок Статус
    return text_of(st.get("preview"), status)


def main():
    cmd = sys.argv[1]
    if cmd == "start":
        chat_id, key, preview, status = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        r = api("sendMessage", chat_id=chat_id, text=text_of(preview, status))
        if not r.get("ok"):
            sys.exit(f"send failed: {r}")
        mid = r["result"]["message_id"]
        json.dump({"chat_id": chat_id, "message_id": mid, "preview": preview.strip()},
                  open(path(key), "w"))
        print(f"started: {key} -> message {mid}")
    elif cmd in ("update", "done"):
        key = sys.argv[2]
        status = sys.argv[3] if cmd == "update" else "Готово"
        st = json.load(open(path(key)))
        r = api("editMessageText", chat_id=st["chat_id"], message_id=st["message_id"],
                text=render(st, status))
        if not r.get("ok") and "message is not modified" not in r.get("description", ""):
            sys.exit(f"edit failed: {r}")
        print(f"updated: {key}")
    else:
        sys.exit("usage: start|update|done")


if __name__ == "__main__":
    main()
