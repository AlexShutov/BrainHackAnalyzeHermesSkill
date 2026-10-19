#!/usr/bin/env python3
"""Clean-install check for the brain-hack-analyze skill (stdlib only, offline).

Clone from the LOCAL repo into a temp dir, build a fresh Hermes-like HOME, run
setup.sh with a mock pip (no network), then run every mode offline: devices-only
index via the BM25 fallback, warm worker protocol, plugin gate, tests on the
clean copy. Nothing writes to the real ~/.hermes.
"""
import json, os, re, shutil, subprocess, sys, tempfile, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable
res, fails = [], []

def check(name, ok, detail=""):
    res.append((name, bool(ok), detail))
    if not ok:
        fails.append(name)

tmp = tempfile.mkdtemp(prefix="bha_clean_")
home = pathlib.Path(tmp) / ".hermes"
home.mkdir(parents=True)
clone = pathlib.Path(tmp) / "clone"

# 1. clone from the local repo
r = subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], capture_output=True, text=True)
check("clone from local repo", r.returncode == 0, r.stderr.strip()[:120])
scripts = clone / "scripts"

# 2. no author absolute paths in runtime code
bad = []
for p in list(scripts.glob("*.py")) + list((clone / "plugin/psych-rag").glob("*.py")):
    txt = p.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"/Users/[A-Za-z0-9_]+/\.hermes", txt):
        bad.append(f"{p.name}: {m.group(0)}")
check("no hardcoded author paths", not bad, "; ".join(bad[:3]))

# 3. setup.sh in DRY_RUN (offline): plan printed, plugin copied into the fresh HOME
env = dict(os.environ, HERMES_HOME=str(home), DRY_RUN="1")
r = subprocess.run(["bash", str(clone / "scripts/setup.sh")], capture_output=True, text=True, env=env, timeout=120)
out = r.stdout + r.stderr
check("setup.sh runs offline in DRY_RUN", r.returncode == 0, (r.stderr.strip()[:150]))
check("venv path derived from HERMES_HOME", str(home / "tools/psych-rag-env") in out, out[:120])
check("install plan printed (network steps named as plan)",
      "would pip install" in out and "fastembed" in out and "would download" in out, out[:160].replace("\n", " "))
check("plugin copied into the fresh HOME", (home / "plugins/psych-rag/__init__.py").exists())
check("config snippet uses generic paths, not the author's", "/Users/" not in out, out[-160:].replace("\n", " "))

# 4. devices-only index in the clean copy via the BM25 fallback (no models, no network)
sys.path.insert(0, str(scripts))
import rag_core as R
con = R.open_db(readonly=False)
R.ensure_fts(con)
devices = json.loads((clone / "psychology/devices.json").read_text(encoding="utf-8"))["devices"]
for d in devices:
    con.execute("INSERT INTO docs(kind,label,text) VALUES(?,?,?)",
                ("device", d["name"], f"{d['family']}: {d['definition']}"))
con.commit()
n = con.execute("SELECT count(*) FROM docs").fetchone()[0]
check("devices-only index builds in the clean copy", n == len(devices), f"{n} docs, {len(devices)} devices")
check("FTS5 index present on the clean copy",
      con.execute("SELECT count(*) FROM sqlite_master WHERE name='docs_fts'").fetchone()[0] == 1)
check("index path lives in the clean copy, not the author's skill dir",
      str(R.DB_PATH).startswith(str(clone)), str(R.DB_PATH))
con.close()
r = subprocess.run([PY, str(scripts / "rag_query.py"), "ложная дилемма", "--no-embed"],
                   capture_output=True, text=True, timeout=60)
check("BM25 fallback query works on the clean copy", r.returncode == 0 and "Ложная дилемма" in r.stdout,
      (r.stdout or r.stderr)[:140].replace("\n", " "))

# 4b. classifier on the clean copy: models are NOT in the repo (model_setup.py creates them).
# Прежний модуль читал labels.txt/thresholds.npy на импорте — навык не деградировал к
# лексикону, а падал целиком (импорт classifier нужен тестам и worker).
import classifier as C
check("classifier imports on the clean copy without models",
      C.LABELS is None and C.THRESH is None,
      f"LABELS present={C.LABELS is not None}, THRESH present={C.THRESH is not None}")
cres = C.classify("мы все знаем, что это правда")
check("classifier answers unavailable on the clean copy",
      isinstance(cres, dict) and cres.get("unavailable"), str(cres)[:120])

# 5. warm worker protocol on the clean copy (no models)
env2 = dict(os.environ, PSYCH_WORKER_NO_MODELS="1", PSYCH_WORKER_IDLE="5")
p = subprocess.Popen([PY, str(scripts / "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, env=env2)
p.stdin.write(json.dumps({"op": "hello"}) + "\n"); p.stdin.flush()
hello = json.loads(p.stdout.readline())
p.stdin.write(json.dumps({"op": "search", "text": "ложная дилемма", "chunk": 1500}) + "\n"); p.stdin.flush()
search = json.loads(p.stdout.readline())
p.stdin.write(json.dumps({"op": "markers", "text": "мы все знаем, что это правда"}) + "\n"); p.stdin.flush()
mk = json.loads(p.stdout.readline())
p.stdin.close(); p.wait(timeout=10)
check("clean-copy worker answers ops via one process", hello.get("ok") and search.get("ok") and mk.get("ok"),
      f"search top={search.get('top')!r}")
check("clean-copy worker exits on EOF (rc 0)", p.returncode == 0, f"rc={p.returncode}")

# 6. tests on the clean copy
r = subprocess.run([PY, str(scripts / "tests/test_layers.py")], capture_output=True, text=True, timeout=300)
check("tests pass on the clean copy", r.returncode == 0,
      (r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-120:]))

# 7. plugin gate constants on the clean copy
plugsrc = (clone / "plugin/psych-rag/__init__.py").read_text(encoding="utf-8")
check("plugin gate constants present on the clean copy",
      "MAX_CHARS = 4096" in plugsrc and "MAX_PARAS = 12" in plugsrc and "RECHECK_SECONDS = 86400" in plugsrc)

print(f"clean install: {len(res) - len(fails)}/{len(res)}")
for name, ok, d in res:
    print(("PASS " if ok else "FAIL ") + name + ((" — " + d) if d else ""))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if fails else 0)
