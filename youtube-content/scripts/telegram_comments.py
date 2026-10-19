#!/usr/bin/env python3
"""Telegram comments under a post — via the LOCAL USER account (Telethon/MTProto).

The Bot API cannot read history or discussion threads; this script uses a
Telethon session stored LOCALLY (per-machine, per-user). The login is lazy:
only when an analysis needs comments and no session exists yet (status
"login_required" — the agent then explains why and walks the user through).

Subcommands:
  status                       — is the session present and authorized?
  login --api-id N --api-hash H --phone +7... [--code C] [--password P]
                               — phase 1 (no code): request the code; phase 2 (with code): finish
  comments --link URL | --chat CHAT --post ID [--limit N]
                               — fetch all comments/replies under a post

Output: one JSON line on stdout. Never prints secrets.
"""
import argparse, json, os, stat, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from util import load_json, read_text

HOME = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("TG_USER_STATE", os.path.expanduser("~/.hermes/tools/telegram-user"))
CONFIG = os.path.join(STATE, "config.json")
SESSION = os.path.join(STATE, "session")
CODEHASH = os.path.join(STATE, "phone_code_hash")


def out(obj):
    print(json.dumps(obj, ensure_ascii=False))


def load_config():
    try:
        return load_json(CONFIG)
    except Exception:
        return None


def client():
    from telethon.sync import TelegramClient
    cfg = load_config()
    return TelegramClient(SESSION, cfg["api_id"], cfg["api_hash"])


def parse_link(url):
    """t.me/<channel>/<id> | t.me/c/<internal_id>/<id> | t.me/s/<channel>/<id>.

    Query и fragment отрезаются до разбора (?t=123 больше не попадает в id).
    rstrip("s") убран — он портил имена каналов, оканчивающиеся на «s»
    («kains» → «kain»); вместо него срезается только сегмент-префикс «s»
    (страница-превью t.me/s/<channel>/<id>).
    Внутренний id канала возвращается числом с маркером -100, а не строкой.
    """
    from urllib.parse import urlsplit
    u = urlsplit(url)
    if "t.me" not in (u.netloc or url):
        return None, None
    parts = [p for p in u.path.strip("/").split("/") if p]
    if parts and parts[0] == "t.me":
        parts = parts[1:]
    if parts and parts[0] == "s":
        parts = parts[1:]
    if len(parts) < 2:
        return None, None
    try:
        post = int(parts[-1])
    except ValueError:
        return None, None
    if parts[0] == "c":
        # внутренний id канала; Telethon требует PeerChannel с ним, а не «-100<id>»
        try:
            return int(parts[1]), post
        except ValueError:
            return None, None
    return parts[-2], post


def peer_for(chat):
    """Telethon-представление канала. «-100<id>» — соглашение Bot API; Telethon
    читает отрицательные id по своей схеме (marked = -(1e12 + id)), поэтому строку
    «-1001543» или int -1001543 он трактует как chat 1001543, а не канал 1543.
    Для канала нужен PeerChannel с внутренним id."""
    from telethon.tl.types import PeerChannel
    if isinstance(chat, int):
        return PeerChannel(chat)
    s = str(chat)
    if s.startswith("-100") and s[4:].isdigit():
        return PeerChannel(int(s[4:]))
    return chat


def cmd_status(_):
    if not load_config() or not os.path.exists(SESSION + ".session"):
        out({"status": "login_required",
             "reason": "нет Telethon-сессии на этой машине",
             "steps": ["api_id/api_hash с my.telegram.org", "телефон", "код из Telegram", "возможен облачный пароль"]})
        return
    with client() as c:
        me = c.get_me()
        out({"status": "ok", "authorized": bool(me),
             "user": (me.first_name or "") if me else None})


def cmd_login(a):
    os.makedirs(STATE, exist_ok=True)
    cfg = {"api_id": a.api_id, "api_hash": a.api_hash, "phone": a.phone}
    with open(CONFIG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    os.chmod(CONFIG, stat.S_IRUSR | stat.S_IWUSR)
    with client() as c:
        if not a.code:
            sent = c.send_code_request(a.phone)
            with open(CODEHASH, "w") as fh:
                fh.write(sent.phone_code_hash)
            out({"status": "code_sent", "hint": "код пришёл в Telegram — повторить login с --code"})
            return
        phone_hash = read_text(CODEHASH).strip() if os.path.exists(CODEHASH) else None
        try:
            c.sign_in(phone=a.phone, code=a.code, phone_code_hash=phone_hash)
        except Exception as e:
            if "password" in str(e).lower() or "SessionPasswordNeeded" in type(e).__name__:
                if not a.password:
                    out({"status": "password_needed", "hint": "повторить login с --code и --password"})
                    return
                c.sign_in(password=a.password)
            else:
                out({"status": "login_failed", "reason": str(e)[:200]})
                return
        if os.path.exists(CODEHASH):
            os.remove(CODEHASH)
        me = c.get_me()
        out({"status": "ok", "authorized": True, "user": (me.first_name or "") if me else None,
             "note": "сессия сохранена локально, повторный логин не нужен"})


def fetch_comments(client, chat, post, limit):
    """Сетевой путь становится тестируемым: клиент передаётся извне, тест подменяет
    его фиксом (mock-transport), и сеть не трогается."""
    entity = client.get_entity(peer_for(chat))
    # limit=0 в Telethon означает «ноль сообщений», а не «все»: iter_messages при
    # limit <= 0 сразу завершается, и путь по умолчанию возвращал пустой список,
    # хотя help обещал «0 = все». «Все» — это None.
    lim = None if limit == 0 else limit
    comments = []
    for m in client.iter_messages(entity, reply_to=post, limit=lim):
        reactions = 0
        if m.reactions:
            reactions = sum(r.count for r in m.reactions.results or [])
        author = m.post_author
        if not author and m.sender is not None:
            author = (getattr(m.sender, "first_name", None)
                      or getattr(m.sender, "title", None)
                      or str(m.sender_id))
        comments.append({
            "id": m.id,
            "reply_to": m.reply_to.reply_to_msg_id if m.reply_to else None,
            "author": author or str(m.sender_id),
            "text": m.text or "",
            "date": m.date.isoformat() if m.date else None,
            "reactions": reactions,
        })
    return comments


def cmd_comments(a):
    chat, post = (parse_link(a.link) if a.link else (a.chat, a.post))
    if not chat or not post:
        out({"status": "bad_input", "reason": "нужны --link или --chat + --post"})
        return
    # dry-run обязан работать БЕЗ сессии: он проверяет разбор ссылки и печатает план,
    # а не требует логина (иначе offline-режим был бы недоступен именно там, где он нужен)
    if getattr(a, "dry_run", False):
        out({"status": "dry_run", "chat": str(chat), "post": post,
             "limit": None if a.limit == 0 else a.limit,
             "would": "get_entity(peer_for(chat)) + iter_messages(reply_to=post)",
             "note": "клиент не создаётся, сеть не трогается"})
        return
    if not load_config() or not os.path.exists(SESSION + ".session"):
        out({"status": "login_required",
             "reason": "нет Telethon-сессии — нужен одноразовый логин под учёткой пользователя",
             "why": "Bot API не умеет читать комментарии; читаем тред обсуждения локальной сессией пользователя"})
        return
    with client() as c:
        comments = fetch_comments(c, chat, post, a.limit)
    result = {"status": "ok", "chat": str(chat), "post": post,
              "count": len(comments), "comments": comments}
    if a.sentiment and comments:
        result["sentiment"] = sentiment_block(comments)
    out(result)


def sentiment_block(comments, use_neural=True):
    """Единый блок анализа комментариев — один входной модуль на всех трёх путях
    (`comment_analysis.py`). Правило навыка: нейро-слой обязателен на каждом анализе;
    раньше Telegram считал только лексиконом, и доставляемые числа расходились в
    зависимости от платформы. Теперь здесь тот же нейро-слой, что и на YouTube, а
    лексикон остаётся кросс-проверкой. Определение треда и «среднего по тредам» —
    те же, что в comment_sentiment (корень + все потомки, простое среднее).

    `use_neural=False` — явный лексикон (для сверки цифр и для тестов)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import comment_analysis as CA
    rows = [{"id": c["id"], "parent": c["reply_to"] or "root",
             "author": c.get("author"), "text": c["text"]} for c in comments]
    res = CA.analyze(rows, use_neural=use_neural)
    tags = {}
    for c in res["comments"]:
        tags[c["tag"]] = tags.get(c["tag"], 0) + 1
    return {"mood": res["mood"], "cross_check": res["cross_check"], "layer": res["layer"],
            "homogeneity_overall": res["overall"], "homogeneity_weighted": res["weighted"],
            "threads": res["threads"], "tags": tags}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    lg = sub.add_parser("login")
    lg.add_argument("--api-id", type=int); lg.add_argument("--api-hash"); lg.add_argument("--phone")
    lg.add_argument("--code"); lg.add_argument("--password")
    cm = sub.add_parser("comments")
    cm.add_argument("--link"); cm.add_argument("--chat"); cm.add_argument("--post", type=int)
    cm.add_argument("--limit", type=int, default=0, help="0 = все")
    cm.add_argument("--sentiment", action="store_true",
                    help="после сбора — настроение и однородность тредов тем же лексиконом, что и comment_sentiment.py")
    cm.add_argument("--dry-run", action="store_true",
                    help="печатает, что сделал бы, не создавая клиент и не трогая сеть")
    a = ap.parse_args()
    {"status": cmd_status, "login": cmd_login, "comments": cmd_comments}[a.cmd](a)


if __name__ == "__main__":
    main()
