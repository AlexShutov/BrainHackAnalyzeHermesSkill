#!/usr/bin/env python3
"""Encrypted book corpus: shipped books live as psychology/corpus/reference_N.enc.

Encryption: the standard `cryptography` package (Fernet = AES-128-CBC + HMAC,
authenticated) — cross-platform wheels for Windows/macOS/Linux, no openssl and
no WSL required. reference_N -> real book name is stored base64-encoded in
REFERENCE_MAP and decoded at runtime. The key is a constant in this module on
purpose: this is obfuscation so the repo does not carry plaintext copyrighted
books, not DRM — the RAG must be able to build itself on a clean machine.

Locally added books (git-ignored raw files dropped into psychology/) are
encrypted the same way by encrypt_new_books(); their mapping goes to
corpus/manifest.local.json (also git-ignored).
"""
import base64, hashlib, json, os, tempfile, unicodedata

from cryptography.fernet import Fernet

ENCRYPT_KEY = "bigbro-corpus-v1::7books::do-not-drop"
_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "psychology", "corpus")
LOCAL_MANIFEST = os.path.join(_CORPUS, "manifest.local.json")
_RAW_EXTS = (".pdf", ".epub", ".doc", ".txt")

def _fernet() -> Fernet:
    digest = hashlib.sha256(ENCRYPT_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))

CORPUS_DIR = _CORPUS

REFERENCE_MAP = {
    "1": "cHN5Y2hvbG9neS/QodC/0LjRgNC40YbQsCDQlS4sINCf0LXQu9C10YXQsNGC0YvQuSDQnC4gLSDQkdC+0LXQstC+0LUg0J3Qm9CfICjQodCw0Lwg0YHQtdCx0LUg0L/RgdC40YXQvtC70L7QsyDQn9C40YLQtdGAKSAtIDIwMjAuZmIy",
    "2": "cHN5Y2hvbG9neS/QndC40LrQuNGC0LAg0J3QtdC/0YDRj9GF0LjQvSAtINCvINC80LDQvdC40L/Rg9C70LjRgNGD0Y4g0YLQvtCx0L7QuS4g0JzQtdGC0L7QtNGLINC/0YDQvtGC0LjQstC+0LTQtdC50YHRgtCy0LjRjyDRgdC60YDRi9GC0L7QvNGDINCy0LvQuNGP0L3QuNGOIC0gMjAxOC5wZGY=",
    "3": "cHN5Y2hvbG9neS/QkdC10YDQvdC10LnRgSDQrdC00LLQsNGA0LQgLSDQn9GA0L7Qv9Cw0LPQsNC90LTQsCAtIDIwMjEuZXB1Yg==",
    "4": "cHN5Y2hvbG9neS/QltCw0Log0K3Qu9C70Y7Qu9GMIC0g0KTQtdC90L7QvNC10L0g0L/RgNC+0L/QsNCz0LDQvdC00YsgKEdhbGxpY2luaXVtKSAtIDIwMjMuZmIy",
    "5": "cHN5Y2hvbG9neS/QrdC60L4g0KPQvNCx0LXRgNGC0L4gLSDQn9C+0LvQvdGL0Lkg0L3QsNC30LDQtCEgWy4uLl0gLSAoQ29ycHVzLiDQktC10YHRjCDQrdC60L4g0KPQvNCx0LXRgNGC0L4pIC0gMjAxMi5mYjI=",
    "6": "cHN5Y2hvbG9neS/QodC10YDQs9C10Lkg0JrQsNGA0LAt0JzRg9GA0LfQsCAtINCc0LDQvdC40L/Rg9C70Y/RhtC40Y8g0YHQvtC30L3QsNC90LjQtdC8LnR4dA==",
    "7": "cHN5Y2hvbG9neS/QkNGA0LXQvdC00YIu0KUu0JjRgdGC0L7QutC4LtGC0L7RgtCw0LvQuNGC0LDRgNC40LfQvNCwLjE5OTYucGRm",
    "8": "cHN5Y2hvbG9neS/Qv9GB0LjRhdC+0LvQvtCz0LjRjyDQvdC70L8g0LTQu9GPINC40LjRiNC60LguZmIy",
}

def _manifest() -> dict:
    """Shipped mapping + local additions (manifest.local.json)."""
    m = dict(REFERENCE_MAP)
    if os.path.exists(LOCAL_MANIFEST):
        try:
            m.update(json.load(open(LOCAL_MANIFEST, encoding="utf-8")))
        except Exception:
            pass
    return m

def real_name(n):
    """reference_N -> real book file name (decoded at runtime)."""
    return base64.b64decode(_manifest()[str(n)]).decode("utf-8")

def encrypt_file(src, dst, n):
    """Encrypt a raw book file into corpus/reference_N.enc."""
    with open(src, "rb") as fi:
        token = _fernet().encrypt(fi.read())
    with open(dst, "wb") as fo:
        fo.write(token)

def decrypt_to_temp(n):
    """Decrypt corpus/reference_N.enc to a temp file with the real extension."""
    raw = os.path.join(CORPUS_DIR, "reference_%d.enc" % n)
    ext = os.path.splitext(real_name(n))[1]
    fd, tmp = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    with open(raw, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(_fernet().decrypt(fi.read()))
    return tmp

def all_references():
    """Sorted absolute paths of every reference_*.enc in the corpus dir."""
    return sorted(
        os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR)
        if f.startswith("reference_") and f.endswith(".enc")
    )

def ref_number(path):
    b = os.path.basename(path)
    return int(b[len("reference_"):-len(".enc")])

def encrypt_new_books() -> int:
    """Encrypt raw books dropped into psychology/ that are not in the corpus yet.

    Raw files stay on disk untouched (git-ignored); only the encrypted copy is
    used downstream. Returns how many new books were encrypted.
    """
    if not os.path.isdir(CORPUS_DIR):
        os.makedirs(CORPUS_DIR, exist_ok=True)
    psy = os.path.dirname(CORPUS_DIR)
    norm = lambda x: unicodedata.normalize("NFC", x)
    known = {norm(os.path.splitext(os.path.basename(real_name(int(n))))[0]) for n in _manifest()}
    def _known(f):
        return norm(os.path.splitext(f)[0]) in known
    used = {ref_number(p) for p in all_references()}
    n_new = 0
    for f in sorted(os.listdir(psy)):
        if not f.endswith(_RAW_EXTS) or f.endswith(".ingest.txt") or _known(f):
            continue
        n = 1
        while n in used:
            n += 1
        src = os.path.join(psy, f)
        dst = os.path.join(CORPUS_DIR, "reference_%d.enc" % n)
        with open(src, "rb") as fi:
            token = _fernet().encrypt(fi.read())
        with open(dst, "wb") as fo:
            fo.write(token)
        m = {}
        if os.path.exists(LOCAL_MANIFEST):
            try:
                m = json.load(open(LOCAL_MANIFEST, encoding="utf-8"))
            except Exception:
                m = {}
        m[str(n)] = base64.b64encode(src.encode("utf-8")).decode("ascii")
        json.dump(m, open(LOCAL_MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        used.add(n)
        n_new += 1
    return n_new
