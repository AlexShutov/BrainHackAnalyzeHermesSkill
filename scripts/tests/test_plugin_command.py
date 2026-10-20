"""Поведенческие тесты плагин-команды /analyze (stdlib, offline, mock-transport).

Каждый тест обязан доказывать расхождение кейсом, где «как есть» и «по правилу» дают
РАЗНЫЙ результат; проверка наличия строки в исходнике — это guard, а не доказанный баг.
Плагин импортируется через spec_from_file_location (package-импорт __init__ не работает),
hermes_cli не тянется. HOME подменяется ДО импорта: модуль решает путь при импорте.
"""
import importlib.util
import os
import tempfile
import unittest

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN_FILE = os.path.join(SKILL_DIR, "plugin", "analyze", "__init__.py")


def load_plugin(home):
    """HOME подменяется до импорта: SCRIPT решается при загрузке модуля."""
    os.environ["HOME"] = home
    os.environ["HERMES_HOME"] = home
    spec = importlib.util.spec_from_file_location("analyze_plugin_under_test", PLUGIN_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeProc(object):
    """Процесс, который ЕЩЁ не завершился: returncode None, wait() возвращает итоговый код."""
    def __init__(self, code=0, stderr=""):
        self.returncode = None
        self._code = code
        self.stderr = stderr

    def wait(self):
        self.returncode = self._code
        return self._code


class PluginCommandTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self._saved_env = {k: os.environ.get(k) for k in ("HOME", "HERMES_HOME")}
        self.mod = load_plugin(self.home)
        self._saved = {"SPAWN": getattr(self.mod, "SPAWN", None),
                       "INJECT": getattr(self.mod, "INJECT", None),
                       "session_key": self.mod.session_key}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(self.mod, k, v)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def make_script(self):
        d = os.path.join(self.home, "scripts")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "analyze.py"), "w") as f:
            f.write("")
        return os.path.join(d, "analyze.py")

    def test_missing_link_says_so(self):
        self.assertIn("нужна ссылка", self.mod.run(""))

    def test_pipeline_path_returns_none_when_script_delivers(self):
        self.make_script()
        calls = []
        self.mod.SPAWN = lambda env, url: calls.append((env, url)) or FakeProc(0)
        self.assertIsNone(self.mod.run("https://youtu.be/X-uxY8XXjWw"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "https://youtu.be/X-uxY8XXjWw")

    def test_running_process_is_not_reported_as_error(self):
        """Расхождение: прежний путь читал returncode у незавершённого процесса (None),
        None != 0 — и сообщал «ОШИБКА» на каждом вызове, даже когда скрипт работал."""
        self.make_script()
        proc = FakeProc(0)
        old_path = "ОШИБКА" if getattr(proc, "returncode", 0) != 0 else None
        self.assertEqual(old_path, "ОШИБКА")                       # прежний путь — ложная ошибка
        self.mod.SPAWN = lambda env, url: proc
        self.assertIsNone(self.mod.run("https://youtu.be/X-uxY8XXjWw"))  # по правилу — тихо

    def test_real_failure_names_stderr(self):
        self.make_script()
        self.mod.SPAWN = lambda env, url: FakeProc(1, "can't open file")
        self.assertIn("can't open file", self.mod.run("https://youtu.be/X-uxY8XXjWw"))

    def test_mockable_spawn_point_is_the_indirection(self):
        """Без подменяемой точки проверки тест сводится к чтению исходника."""
        self.make_script()
        hit = []
        self.mod.SPAWN = lambda env, url: hit.append(url) or FakeProc(0)
        self.mod.run("https://youtu.be/X-uxY8XXjWw")
        self.assertEqual(hit, ["https://youtu.be/X-uxY8XXjWw"])

    def test_script_path_resolved_from_hermes_home_not_author(self):
        got = self.mod.script_candidates()[0]
        self.assertTrue(got.startswith(self.home), got)
        self.assertNotIn("/Users/alex/.hermes/skills", got)

    # -- ссылка обязана дойти до навыка -----------------------------------
    def test_link_extracted_from_args_not_read_as_whole(self):
        got = self.mod.extract_url("Анализ https://www.youtube.com/watch?v=X-uxY8XXjWw с таймкодами")
        self.assertEqual(got, "https://www.youtube.com/watch?v=X-uxY8XXjWw")

    def test_raw_video_id_becomes_link(self):
        self.assertEqual(self.mod.extract_url("X-uxY8XXjWw"),
                         "https://www.youtube.com/watch?v=X-uxY8XXjWw")

    def test_link_reaches_the_session_when_pipeline_absent(self):
        """Расхождение: без конвейера прежний обработчик возвращал None — dispatch
        плагин-команды идёт до хода агента, и None означает «ничего не доставлено»,
        то есть навык ссылку не увидел бы вообще."""
        seen = []
        self.mod.session_key = lambda: "agent:main:telegram:dm:362122320"

        def inject(content, role, session_key=None):
            seen.append((content, role, session_key))
            return True

        self.assertIsNone(self.mod.run("https://www.youtube.com/watch?v=X-uxY8XXjWw", inject=inject))
        self.assertEqual(len(seen), 1)
        content, role, key = seen[0]
        self.assertIn("X-uxY8XXjWw", content)
        self.assertEqual(role, "user")
        self.assertEqual(key, "agent:main:telegram:dm:362122320")

    def test_denied_injection_is_reported_not_swallowed(self):
        self.mod.session_key = lambda: "agent:main:telegram:dm:362122320"
        out = self.mod.run("https://youtu.be/X-uxY8XXjWw", inject=lambda *a, **k: False)
        self.assertIn("не получил ссылку", out)
        self.assertIn("allow_gateway_injection", out)

    def test_missing_session_key_is_reported(self):
        self.mod.session_key = lambda: ""
        out = self.mod.run("https://youtu.be/X-uxY8XXjWw", inject=lambda *a, **k: True)
        self.assertIn("нет ключа сессии", out)

    def test_no_pipeline_and_no_agent_says_so(self):
        out = self.mod.run("https://youtu.be/X-uxY8XXjWw", inject=None, ctx=None)
        self.assertIn("ход агента недоступен", out)

    def test_source_has_no_absolute_author_path(self):
        with open(PLUGIN_FILE) as f:
            src = f.read()
        self.assertNotIn("/Users/alex", src)


if __name__ == "__main__":
    unittest.main()
