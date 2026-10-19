#!/usr/bin/env python3
"""Тесты кода навыка brain-hack-analyze (unittest, stdlib).

Запуск:
  python3 scripts/tests/test_layers.py
  /Users/alex/.hermes/tools/psych-rag-env/bin/python scripts/tests/test_layers.py

Модели подменяются фиксами, где это нужно для логики, — тесты не качают HF и
не грузят ONNX. Два блока:
  A. регрессия на уже исправленном (должны проходить);
  B. проверки, кодирующие правила навыка, которые кодом НЕ реализованы.
     Они падают — это и есть остаток багов; тест их не «чинит», он показывает,
     где правило расходится с кодом. Такие тесты помечены в docstring.
"""
import contextlib, importlib.util, io, json, os, re, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
SKILL = os.path.dirname(SCRIPTS)
YT = os.path.join(SKILL, "youtube-content", "scripts")
PLUGIN = os.path.join(SKILL, "plugin", "psych-rag", "__init__.py")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, YT)

import markers, side
import types
import ukr_detect as ukr, comment_sentiment, comment_analysis as CA, status_post, neural_pass
import classifier, rag_core as R

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

try:
    import telegram_comments as tele
except Exception:
    tele = None

def load_plugin():
    spec = importlib.util.spec_from_file_location("psychrag", PLUGIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# --- фиксы для тестов логики без моделей -------------------------------------
WORD2ID, ID2WORD = {}, {}
def word_id(w):
    if w not in WORD2ID:
        WORD2ID[w] = len(WORD2ID) + 10
        ID2WORD[WORD2ID[w]] = w
    return WORD2ID[w]

class Enc:
    def __init__(self, ids): self.ids, self.attention_mask = ids, [1] * len(ids)

class FakeTok:
    def encode(self, text, add_special_tokens=True):
        words = re.findall(r"\w+", text)
        return Enc([1] + [word_id(w) for w in words] + [2])

TARGET = "подмена"

class FakeSession:
    """Фикс вместо ONNX: метка даётся только на том куске, где есть TARGET-слово,
    поэтому тест может проверить, что классификатор реально видит хвост абзаца."""
    def run(self, _, feeds):
        ids = feeds["input_ids"][0]
        words = {ID2WORD.get(int(i), "") for i in ids}
        logits = np.full(len(classifier.LABELS), -3.0)
        if TARGET in words:
            logits[classifier.LABELS.index("Doubt")] = 3.0
        return [np.array([logits])]

def run_classifier(text, extra=()):
    argv = sys.argv
    sys.argv = ["classifier.py", text, "--json"] + list(extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        classifier.main()
    sys.argv = argv
    return json.loads(buf.getvalue())

# --- A. регрессия на исправленном --------------------------------------------
class TestFixed(unittest.TestCase):
    def setUp(self):
        classifier.session = lambda: FakeSession()
        classifier.tokenizer = lambda: FakeTok()

    def test_hit_carries_paragraph_index(self):
        text = "читатель видит только цену и не видит качество товара\n\nтот же автор говорит про цену и качество и снова не видит проблему"
        hits = markers.scan(text)
        rep = [h for h in hits if h["marker"] == "repetition"]
        self.assertTrue(rep, "repetition-hit должен быть")
        self.assertEqual(rep[0]["para"], [1, 2], "repetition обязан нести оба абзаца")

    def test_gate_keeps_repetition_only_paragraph(self):
        # абзац 1 помечен только повтором (сам по себе он ничего не даёт классификатору),
        # TARGET-слово — в нём, чтобы было видно, прошёл ли он гейт
        text = "читатель видит только цену и не видит качество товара подмена\n\nтот же автор говорит про цену и качество и снова не видит проблему"
        rows = run_classifier(text, ["--gated"])["classification"]
        self.assertIn(1, [r["para"] for r in rows],
                      "абзац, помеченный только повтором, должен проходить гейт")

    def test_long_paragraph_is_chunked_not_cut(self):
        para = ("слово " * 300) + "подмена"
        out = classifier.classify(para, 0.70)
        self.assertTrue(out, "длинный абзац должен пройти целиком")
        self.assertGreater(out[0]["chunk"], 1, "метка должна прийти из хвостового куска")
        self.assertEqual(classifier.classify("слово " * 300, 0.70), [],
                         "без приёма метки быть не должно")

    def test_window_positions_match_original_text(self):
        # «İ» при lower() даёт два символа («i̇»): позиция из lower-строки съезжала
        # на один символ, и window() начинался на символ позже фактического совпадения
        text = "İ " + "x" * 40 + " та " + "y" * 40
        h = markers.scan(text)
        self.assertTrue(h, "маркер «та» должен быть найден")
        self.assertTrue(h[0]["context"].startswith("x" * 29 + " та"),
                        "context обязан считаться от позиции в исходной строке, "
                        f"получено: {h[0]['context'][:35]!r}")

    def test_telegram_limit_zero_is_none(self):
        if tele is None:
            self.skipTest("telethon не установлен")
        seen = {}
        class FakeClient:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get_entity(self, e): return e
            def iter_messages(self, entity, reply_to=None, limit=None):
                seen["limit"] = limit
                return []
        tele.client = lambda: FakeClient()
        tele.load_config = lambda: {"api_id": 1, "api_hash": "x"}
        tele.os.path.exists = lambda p: True
        tele.out = lambda o: None
        class A: link, chat, post, limit, sentiment = "t.me/x/1", None, None, 0, False
        tele.cmd_comments(A())
        self.assertIsNone(seen.get("limit"), "0 должно превращаться в None («все»)")
        A.limit = 50
        tele.cmd_comments(A())
        self.assertEqual(seen.get("limit"), 50)

    def test_lexicon_word_boundary_and_negation(self):
        self.assertEqual(comment_sentiment.classify("ураган")[0], "нейтрально",
                         "короткий стем «ура» ловится только как целое слово")
        self.assertEqual(comment_sentiment.classify("ура")[0], "радость")
        self.assertEqual(comment_sentiment.classify("классический фильм")[0], "нейтрально",
                         "«класс» убран из лексикона")
        self.assertEqual(comment_sentiment.classify("не лучший")[0], "недовольство",
                         "отрицание переворачивает позитив")
        self.assertEqual(comment_sentiment.classify("мне надо")[0], "нейтрально",
                         "«мне» не должно читаться как «не надо»")

    def test_ukr_detect_dictionary_and_boundaries(self):
        self.assertEqual(ukr.lang_of("Копам він пояснив, що знайшов предмет"), "uk")
        self.assertEqual(ukr.lang_of("Гэта быў вельмі interesting момант"), "be")
        self.assertEqual(ukr.lang_of("Мы обсуждали закон и выборы как факты"), "ru")
        self.assertEqual(ukr.word_hit("щелей", {"ще"}), False,
                         "короткая запись (≤3) ловится только как целое слово")
        self.assertEqual(ukr.lang_of("Як але і де"), "amb",
                         "общие uk/be слова язык не решают")
        for stem in ("район", "станц", "курсу", "уходу", "електротранспорт"):
            self.assertNotIn(stem, ukr.UK_WORDS,
                             f"стем «{stem}» совпадает с русским — в словаре его быть не должно")

    def test_rag_query_is_chunked_by_callers(self):
        """Поведение: все три пути реально делят длинный запрос на куски — проверяется
        фактическим вызовом (spy), а не словом "chunk" в исходнике."""
        long_text = "подмена " + "слово " * 400            # > 1500 -> несколько кусков
        hit = {"results": [{"kind": "device", "label": "X", "score": 0.5, "snippet": "y"}]}
        p = load_plugin()
        calls = []
        p.worker_call = lambda op, **kw: (calls.append((op, kw)), hit)[1]
        p.retrieve(long_text)
        search = [kw for op, kw in calls if op == "search"]
        self.assertTrue(search, "retrieve обязан идти через worker_call")
        self.assertGreater(len(search[0]["chunks"]), 1, "плагин обязан делить длинный запрос")

        import psych_layer as PL
        calls.clear()
        PL.worker_call = lambda op, **kw: (calls.append((op, kw)),
                                           {"markers": [], "classification": [], "results": [hit["results"][0]]})[1]
        f = os.path.join(tempfile.mkdtemp(), "t.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(long_text)
        argv, sys.argv = sys.argv, ["psych_layer.py", "--file", f]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                PL.main()
        finally:
            sys.argv = argv
        search = [kw for op, kw in calls if op == "search"]
        self.assertTrue(search, "psych_layer обязан опрашивать индекс через worker_call")
        self.assertGreater(len(search[0]["chunks"]), 1,
                           "psych_layer обязан опрашивать индекс кусками, а не срезом первых 1500")

        try:
            import rag_mcp
        except Exception:
            self.skipTest("MCP-модуль не импортируется (нет fastembed)")
        seen = {}
        def chunks_of(q):
            seen["chunks"] = [q[:1500], q[1500:]]
            return seen["chunks"]
        def merge_chunked(chunks, top_k=6, qvecs=None):
            seen["used"] = len(chunks)
            return []
        rag_mcp.R = types.SimpleNamespace(TOP_K=6, chunks_of=chunks_of, merge_chunked=merge_chunked)
        rag_mcp.get_model = lambda: False
        res = rag_mcp.handle({"method": "tools/call", "params": {"name": rag_mcp.TOOL["name"],
                                                                 "arguments": {"query": long_text}}})
        self.assertIn("chunks", seen, "MCP-путь обязан вызывать chunk на длинном запросе")
        self.assertGreater(seen.get("used", 0), 1, "MCP обязан искать по всем кускам")
        self.assertIn("ничего не найдено", res["content"][0]["text"])

# --- B. правила, которые кодом не реализованы (падают) -----------------------
class TestOpenBugs(unittest.TestCase):
    def test_plugin_gate_equals_platform_limit(self):
        """Поведение: сообщение 4061 символа с 2 абзацами проходит гейт, 4121 — отклоняется;
        итог никогда не превышает лимит платформы."""
        p = load_plugin()
        p.worker_call = lambda op, **kw: None
        p.retrieve = lambda t: None
        p.marker_hits = lambda t: [{"family": "f", "marker": "m", "found": "x", "context": "y"}]
        p.classifier_hits = lambda t, h: None
        p.version_check = lambda s: None
        class Ev:
            def __init__(self, text):
                self.text, self.internal, self.source = text, False, None
        base = "подмена " * 500                                  # 4000 символов
        out = p.callback(event=Ev(base + "\n\n" + "х" * 59))    # 4061
        self.assertIsNotNone(out, "сообщение <= 4096 обязано проходить гейт")
        self.assertLessEqual(len(out["text"]), p.MAX_CHARS, "итог не должен превышать лимит платформы")
        self.assertEqual(p.MAX_CHARS, 4096, "гейт обязан равняться фактическому лимиту платформы")
        self.assertIsNone(p.callback(event=Ev(base + "\n\n" + "х" * 119)),   # 4121
                          "сообщение > 4096 обязано отклоняться")

    def test_proper_name_alone_is_not_pig(self):
        """Правило: имя в украинской орфографии — не мова-вставка, 🐷 не ставится."""
        lang, segs = ukr.detect_item("Євгененко вышел из дома и пошёл к машине.")
        tag = "🐷" if any(sl == "uk" for sl, _ in segs) else None
        self.assertIsNone(tag, "единственное доказательство — имя: 🐷 не ставится")

    def test_short_fragment_does_not_decide_language(self):
        """Правило: фрагменты ≤3 слов язык не решают."""
        self.assertEqual(ukr.lang_of("Копам він"), "amb", "двухсловный фрагмент не решает")

    def test_majority_counts_only_decided_languages(self):
        """Правило: доля не-русского считается по решённым языкам, amb — не язык."""
        _, share_chars, share_items = ukr.majority(
            ["Мы обсуждали закон и выборы как факты." * 20, "Як але і де — язык не решаем."])
        self.assertEqual(share_chars, 0.0, "amb не должен входить в долю не-русского")

    def test_side_end_map_rules_are_reachable(self):
        """Правило END_MAP должно быть достижимо фактическим классом слов."""
        n = int(re.search(r"\{(\d+),\}", side.SIDE_CLASS).group(1))
        for src, _ in side.END_MAP:
            self.assertGreaterEqual(len(src), n, f"правило «{src}» недостижимо при классе {{{n},}}")
        self.assertEqual(side.to_ru("манипуляцыя"), "манипуляция",
                         "правило должно применяться на живом слове")

    def test_side_share_has_a_threshold(self):
        self.assertTrue(hasattr(side, "SIDE_THRESHOLD"),
                        "docstring обещает «≥ порога», порога в коде нет")

    def test_side_share_and_ukr_detect_agree_on_short_fragment(self):
        """Два слоя совпадают в классе, а не в имени языка: если боковой слой (графика)
        видит, что фрагмент весь на мове, морфология не называет его русским. Конкретный
        язык морфология решает только по решающей букве — общая 'і' язык не решает,
        поэтому на «маніпуляцыя» вердикт 'amb', а не 'uk'."""
        thr = side.SIDE_THRESHOLD
        frag = "маніпуляцыя"
        self.assertGreaterEqual(side.side_share(frag), thr)
        self.assertNotEqual(ukr.lang_of(frag), "ru",
                            "боковой слой видит мову, морфология называет её русским")
        frag2 = "манипуляция"
        self.assertLess(side.side_share(frag2), thr)
        self.assertEqual(ukr.lang_of(frag2), "ru",
                         "ниже порога — это вставка, а не язык текста")

    def test_status_post_guard_length_and_flood(self):
        """Поведение: длинный статус усекается до лимита, превью сохраняется целиком;
        429 повторяется, а не роняет скрипт."""
        preview = "https://youtu.be/abc " + "заголовок " * 100
        status = "статус " * 2000
        t = status_post.clamp(preview, status)
        self.assertLessEqual(len(t), status_post.TG_LIMIT)
        self.assertTrue(t.startswith(preview.strip()), "превью обязано сохраниться целиком")
        self.assertIn("Статус:", t)

        import urllib.error
        attempts = {"n": 0}
        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"ok": true}'
        err = urllib.error.HTTPError("u", 429, "Too Many Requests", {},
                                     io.BytesIO(b'{"ok":false,"parameters":{"retry_after":1}}'))
        def fake_urlopen(req, timeout=15):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise err
            return Resp()
        orig = (status_post.urllib.request.urlopen, status_post.bot_token, status_post.time.sleep)
        status_post.urllib.request.urlopen = fake_urlopen
        status_post.bot_token = lambda: "T"
        status_post.time.sleep = lambda s: None
        try:
            r = status_post.api("sendMessage")
            self.assertTrue(r.get("ok"), "429 обязана быть повторена, а не убить скрипт")
            self.assertEqual(attempts["n"], 2, "первая попытка — 429, вторая — успех")
        finally:
            err.close()
            (status_post.urllib.request.urlopen, status_post.bot_token, status_post.time.sleep) = orig

    def test_tie_order_is_one_for_both_paths(self):
        """Один лексикон — один порядок ничьи. Проверка поведения, а не строки в файле:
        telegram-путь не держит свой словарь и обязан совпадать с comment_sentiment."""
        if tele is None:
            self.skipTest("telethon не установлен")
        text = "злой и плохо"                      # ничья: злость и недовольство по одному
        lex = comment_sentiment.classify(text)[0]
        self.assertEqual(lex, "злость", "ничья в classify решается по TIE_ORDER")
        comments = [{"id": "R1", "reply_to": None, "text": text},
                    {"id": "R1.b1", "reply_to": "R1", "text": text}]
        block = tele.sentiment_block(comments, use_neural=False)
        rows = [{"id": c["id"], "parent": c["reply_to"] or "root", "text": c["text"],
                 "author": None, "cat": comment_sentiment.classify(c["text"])[0]}
                for c in comments]
        th, by = comment_sentiment.build_threads(rows)
        lex_dom = comment_sentiment.thread_homogeneity(th, by)[0][0]["dominant"]
        self.assertEqual(block["threads"][0]["dominant"], lex_dom,
                         "ничья в треде на telegram-пути решается не по TIE_ORDER")
        self.assertEqual(block["layer"], "lexicon",
                         "telegram-путь не идёт через общий модуль анализа")

    def test_neural_labels_have_no_none_and_known_models(self):
        """Поведение: пропуск в id2label не даёт None-метку (модель подменяется фиксом)."""
        class FakeCfg:
            id2label = {0: "anger", 1: None, 2: "joy"}
        class FakeModel:
            def load_state_dict(self, sd): pass
            def eval(self): return self
        neural_pass.hf_hub_download = lambda repo, f: "/tmp/x"
        neural_pass.AutoConfig = type("A", (), {"from_pretrained": staticmethod(lambda r: FakeCfg())})
        neural_pass.AutoModelForSequenceClassification = type("B", (),
                                                              {"from_config": staticmethod(lambda c: FakeModel())})
        neural_pass.load_bytes = lambda b: {}
        neural_pass.read_bytes = lambda p: b""
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            model, labels = neural_pass.load_no_mmap("seara/fake")
        self.assertNotIn(None, labels, "None-метка теряла вердикт по этой модели")
        self.assertEqual(labels, ["anger", "class_1", "joy"],
                         "пропуск обязан становиться явным class_N, а не молча терять вердикт")
        self.assertTrue(set(neural_pass.MULTILABEL) >= set(neural_pass.REPOS),
                        "fallback-порядок есть не для всех моделей индекса")

    # --- новое, найденное этим проходом ---
    def test_runtime_classifier_operating_point(self):
        """Поведение: порог = max(thresholds.npy, min_conf). Фикс ONNX даёт Doubt с p=0.60
        (её порог 0.47 ниже 0.70) и Appeal_to_Time с p=0.80 (порог 0.66)."""
        class FixedSession:
            def __init__(self, probs): self.probs = probs
            def run(self, _, feeds):
                logits = np.full(len(classifier.LABELS), -3.0)
                for name, p in self.probs.items():
                    logits[classifier.LABELS.index(name)] = np.log(p / (1 - p))
                return [np.array([logits])]
        classifier.session = lambda: FixedSession({"Doubt": 0.60, "Appeal_to_Time": 0.80})
        classifier.tokenizer = lambda: FakeTok()
        plain = {r["label"] for r in classifier.classify("абзац")}
        self.assertIn("Doubt", plain, "без min_conf порога 0.47 достаточно p=0.60")
        strict = {r["label"] for r in classifier.classify("абзац", min_conf=0.70)}
        self.assertNotIn("Doubt", strict, "min_conf=0.70 обязан отсеять p=0.60")
        self.assertIn("Appeal_to_Time", strict, "p=0.80 обязана остаться")
        # плагин обязан передавать рабочую точку — проверяется фактическим аргументом вызова
        p = load_plugin()
        calls = []
        p.worker_call = lambda op, **kw: (calls.append((op, kw)),
                                          {"classification": [{"para": 1, "labels": []}]})[1]
        p.classifier_hits("абзац", [{"marker": "m"}])
        self.assertEqual(calls[0][1].get("min"), 0.70, "плагин зовёт worker без рабочей точки")

    def test_thread_definition_is_one_across_paths(self):
        if tele is None:
            self.skipTest("telethon не установлен")
        rows = [{"id": "R1", "parent": "root", "text": "злость"},
                {"id": "R1.b1", "parent": "R1", "text": "радость"},
                {"id": "R1.b1.c1", "parent": "R1.b1", "text": "радость"}]
        for r in rows:
            r["cat"] = comment_sentiment.classify(r["text"])[0]
        th, by = comment_sentiment.build_threads(rows)
        lex_hom = comment_sentiment.thread_homogeneity(th, by)[1]
        comments = [{"id": r["id"], "reply_to": None if r["parent"] == "root" else r["parent"],
                     "text": r["text"]} for r in rows]
        block = tele.sentiment_block(comments, use_neural=False)
        self.assertEqual(block["homogeneity_overall"], lex_hom,
                         f"telegram-путь даёт {block['homogeneity_overall']}% против {lex_hom}% "
                         "— вложенный ответ открывает отдельный тред")

    def test_overall_homogeneity_is_one_definition(self):
        """Поведение: «в среднем по тредам» — простое среднее по измеримым тредам, не
        взвешенное. Тред A: 3 содержательных (100%), тред B: 2 (50%) -> простое 75,
        взвешенное по размеру 80. Все пути идут через comment_analysis."""
        rows = [{"id": "R1", "author": "a", "text": "злой и злой", "parent": "root", "likes": 0},
                {"id": "R1.b1", "author": "b", "text": "злой и злой", "parent": "R1", "likes": 0},
                {"id": "R1.b2", "author": "c", "text": "злой и злой", "parent": "R1", "likes": 0},
                {"id": "R2", "author": "d", "text": "радостно и радостно", "parent": "root", "likes": 0},
                {"id": "R2.b1", "author": "e", "text": "злой и злой", "parent": "R2", "likes": 0}]
        rep = CA.analyze(rows, use_neural=False)
        self.assertEqual(rep["overall"], 75, "среднее по тредам обязано быть простым")
        self.assertEqual(rep["weighted"], 80, "взвешенное обязано считаться отдельно")
        for name in ("rutube_comments.py", "telegram_comments.py", "comment_sentiment.py"):
            self.assertIn("comment_analysis", read(os.path.join(YT, name)),
                          f"{name} строит свой тред/среднее вместо общего модуля")

    def test_lexicon_has_no_duplicate_stems(self):
        from collections import Counter
        for cat, stems in comment_sentiment.CATS.items():
            dups = [s for s, n in Counter(stems).items() if n > 1]
            self.assertFalse(dups, f"{cat}: дубликат {dups} — одно и то же слово считается дважды")

    def test_stop_list_is_effective_for_repetition(self):
        """Стоп-лист работает только если слово может быть выдано классом значимых слов —
        проверяется по фактическим константам markers, а не по копии списка в тесте."""
        n = int(re.search(r"\{(\d+),\}", markers.SIGNIFICANT_CLASS).group(1))
        unreachable = [w for w in markers.STOP if len(w) < n]
        self.assertFalse(unreachable,
                         f"{len(unreachable)} из {len(markers.STOP)} стоп-слов недостижимы при {{{n},}}")
        for w in ("очень", "просто", "также", "именно"):
            self.assertIn(w, markers.STOP,
                          f"«{w}» — служебное слово достижимого класса, которого в списке нет")

    def test_ambig_and_weak_are_used_in_the_decision(self):
        """docstring: неоднозначные считаются вместе с маркером; омонимы усиливают решённый язык."""
        # поведение: неоднозначные без маркера — не русский; одна омонимичная форма язык не решает
        self.assertEqual(ukr.lang_of("це ми бачили де так"), "amb",
                         "неоднозначные слова без маркера уходили в ru")
        self.assertEqual(ukr.lang_of("па был в доме"), "ru",
                         "одна омонимичная форма не должна решать язык")

    def test_hf_language_filter_accepts_prefixed_tags(self):
        """Поведение: кандидат с тегом `language:ru` проходит фильтр, с `language:en` — нет."""
        import model_update_check as M
        def fake_hub(url):
            if "api/models/cur" in url:
                return {"model_id": "cur", "sha": "r1",
                        "model-index": [{"results": [{"metrics": [{"type": "f1", "value": 0.45, "name": "macro"}]}]}]}
            return [{"id": "newru", "sha": "s2", "tags": ["language:ru"],
                     "model-index": [{"results": [{"metrics": [{"type": "f1", "value": 0.60, "name": "macro"}]}]}]},
                    {"id": "newen", "sha": "s3", "tags": ["language:en"],
                     "model-index": [{"results": [{"metrics": [{"type": "f1", "value": 0.70, "name": "macro"}]}]}]}]
        M.load_version = lambda: {"model_id": "cur", "revision": "r1"}
        M.load_state = lambda: None
        M.hub_get = fake_hub
        argv, sys.argv = sys.argv, ["model_update_check.py", "--dry-run", "--json"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                M.main()
        finally:
            sys.argv = argv
        out = json.loads(buf.getvalue())
        self.assertEqual(out["state"]["candidate"]["model_id"], "newru",
                         "фильтр обязан ловить `language:ru`, а не только 2-буквенный тег")

    def test_fetch_transcript_outputs_promised_language_key(self):
        """Поведение: main печатает обещанные docstring-ем поля language и generated."""
        import fetch_transcript as FT
        segs = [{"text": "привет", "start": 0.0, "duration": 1.0}]
        FT.fetch_transcript = lambda vid, langs=None: (segs, "ru", True)
        argv, sys.argv = sys.argv, ["fetch_transcript.py", "dQw4w9WgXcQ"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                FT.main()
        finally:
            sys.argv = argv
        out = json.loads(buf.getvalue())
        self.assertEqual(out.get("language"), "ru", "docstring обещает поле language в выводе")
        self.assertTrue(out.get("generated"), "авто-трек обязан быть отмечен")

    def test_rag_mcp_chunks_long_query(self):
        """Строковая проверка заменена: chunk-путь MCP проверяется поведением в
        test_rag_query_is_chunked_by_callers (spy фактического вызова)."""

    # --- новое, найденное этим проходом (тесты добавлены вместе с фиксом) ------
    def test_classifier_chunks_overlap(self):
        """Приём на стыке двух окон 256 токенов терялся — куски обязаны перекрываться."""
        starts = classifier.chunk_starts(600)
        self.assertGreater(len(starts), 1)
        self.assertTrue(all(b - a < classifier.MAX_TOKENS for a, b in zip(starts, starts[1:])),
                        "шаг равен окну без перекрытия: приём на стыке не попадает ни в одно окно")

    def test_min_ev_branch_is_live(self):
        """min_ev обязан что-то решать: при min_ev=2 одного доказательства мало."""
        frag = "він пояснив предмет быстро"
        self.assertEqual(ukr.lang_of(frag), "uk")
        self.assertEqual(ukr.lang_of(frag, min_ev=2), "amb",
                         "ветка min_ev мертва: порог никогда не применяется")

    def test_plugin_guard_covers_all_markers(self):
        """Guard обязан покрывать все четыре блока, не только первый."""
        p = load_plugin()
        p.retrieve = lambda t: [{"kind": "device", "label": "X", "score": 0.5, "snippet": "y"}]
        p.marker_hits = lambda t: None
        p.classifier_hits = lambda t, h: None
        p.version_check = lambda s: None
        class Ev:
            def __init__(self, text):
                self.text, self.internal = text, False
        already = "абзац один\n\nабзац два" + p.MARK2 + "\n[структурные маркеры] / повтор: [x] y"
        self.assertIsNone(p.callback(event=Ev(already)),
                          "сообщение, которое уже содержит блок маркеров, переписывается повторно")
        self.assertIsNotNone(p.callback(event=Ev("абзац один\n\nабзац два")),
                            "контроль: без маркера переписывание работает")

    def test_readonly_db_and_closed_files(self):
        """Поведение: запросное соединение отказывает при попытке записи (режим ro)."""
        import sqlite3
        db = os.path.join(tempfile.mkdtemp(), "index.db")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, kind TEXT, label TEXT, text TEXT)")
        con.commit(); con.close()
        orig = R.DB_PATH
        R.DB_PATH = db
        try:
            ro = R.open_db(readonly=True)
            try:
                self.assertRaises(sqlite3.OperationalError, ro.execute,
                                  "INSERT INTO docs VALUES (1,'a','b','c')",
                                  )
            finally:
                ro.close()
        finally:
            R.DB_PATH = orig
        for f in sorted(os.listdir(SCRIPTS)):
            if f.endswith(".py") and f != "util.py":   # util.py — сам помощник, в его docstring шаблон упомянут намеренно
                s = read(os.path.join(SCRIPTS, f))
                self.assertNotIn("json.load(open(", s, f"{f}: незакрытый open()")

    def test_status_post_argv_guard(self):
        """Поведение: неполная команда даёт usage, а не IndexError."""
        argv, sys.argv = sys.argv, ["status_post.py", "start", "1"]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                status_post.main()
        except SystemExit as e:
            self.assertIn("usage", str(e).lower())
        else:
            self.fail("неполная команда обязана завершаться usage-сообщением")

class TagRuleTests(unittest.TestCase):
    """Поведенческие тесты на починки тегов комментария (пункты аудита)."""

    def test_toxic_tag_respects_threshold(self):
        # было: любое ненулевое insult -> 🤢 (p=0.01 — не токсичность)
        self.assertEqual(CA.tag("обычный текст", "нейтрально", {"toxicity": {"insult": 0.01}}), "🙂")
        self.assertEqual(CA.tag("обычный текст", "нейтрально", {"toxicity": {"insult": 0.35}}), "🤢")
        self.assertEqual(CA.tag("обычный текст", "нейтрально", {"toxicity": {"insult": 0.80}}), "🤮")
        self.assertEqual(CA.tag("обычный текст", "нейтрально",
                                {"toxicity": {"insult": 0.35, "obscenity": 0.32}}), "🤮")
        self.assertEqual(CA.tag("обычный текст", "нейтрально",
                                {"toxicity": {}, "sarcasm": {"sarcasm": 0.29}}), "🙂")
        self.assertEqual(CA.tag("обычный текст", "нейтрально",
                                {"toxicity": {}, "sarcasm": {"sarcasm": 0.31}}), "🤢")
        self.assertIn("🤢", CA.PRIORITY, "константа приоритета расходится с фактическим порядком tag()")

    def test_death_tag_is_human_only(self):
        self.assertIsNone(CA.death_tag("мотор умер на холостых"), "смерть техники — не ☠️")
        self.assertIsNone(CA.death_tag("кошка погибла под машиной"), "смерть животного — не ☠️")
        self.assertIsNone(CA.death_tag("погиб"), "без человека-субъекта метка не ставится")
        self.assertEqual(CA.death_tag("человек погиб"), "☠️")
        self.assertEqual(CA.death_tag("умерла мама"), "☠️")
        self.assertEqual(CA.death_tag("это суицид"), "☠️")
        # приоритет: 💸 -> приём -> ☠️ -> 🔴
        self.assertEqual(CA.tag("купите, человек погиб", "недовольство", {}), "💸")
        self.assertEqual(CA.tag("человек погиб", "недовольство", {}), "☠️")

    def test_priority_order_matches_docstring(self):
        """Docstring — это доставляемое правило; он обязан совпадать с фактическим
        порядком tag(). Раньше в нём не было 🤢, и правило обещало 🔴 там, где код
        ставит 🤢. Проверка поведением: комментарий, где применимы и 🤢, и 🔴
        (недовольство + insult 0.35), — код выбирает 🤢, и docstring обязан это отражать."""
        both = CA.tag("обычный текст", "недовольство", {"toxicity": {"insult": 0.35}})
        self.assertEqual(both, "🤢", "фактический порядок: 🤢 перед 🔴")
        line = [l for l in CA.tag.__doc__.splitlines() if "→" in l][0]
        stated = [s.strip() for s in line.split("→")]
        self.assertEqual(stated, CA.PRIORITY, "docstring приоритета расходится с PRIORITY")
        # и каждый соседний шаг порядка проверяется поведением, а не строкой
        self.assertEqual(CA.tag("купите по ссылке", "недовольство",
                                {"toxicity": {"insult": 0.80}}), "💸")
        self.assertEqual(CA.tag("человек погиб", "недовольство",
                                {"toxicity": {"insult": 0.80}}), "☠️")
        self.assertEqual(CA.tag("обычный текст", "радость", {}), "😊")
        self.assertEqual(CA.tag("обычный текст", "нейтрально", {}), "🙂")


class RagPathTests(unittest.TestCase):
    """Поведенческие тесты RAG: FTS5-путь, fallback, резерв устройств, одиночный meta-запрос,
    read-only соединение. Проверяется поведение, а не наличие слов в исходнике."""

    ROWS = [("device", "Ложная дилемма", "только два варианта: либо A, либо B"),
            ("device", "Ошибка выжившего", "я заработал, значит метод работает"),
            ("chunk", "Книга — Глава 1", "автор говорит: либо мы, либо они — это ложная дилемма выбора"),
            ("chunk", "Книга — Глава 2", "гарантированная доходность без риска противоречит теории")]

    def _mem(self, with_fts=True):
        import sqlite3
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, kind TEXT, label TEXT, text TEXT)")
        con.executemany("INSERT INTO docs(kind, label, text) VALUES (?,?,?)", self.ROWS)
        if with_fts:
            R.ensure_fts(con)
            con.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
        return con

    def test_fts5_and_scan_agree_on_top_hit(self):
        con = self._mem(with_fts=True)
        q = "ложная дилемма либо мы либо они"
        fts = R.bm25_scores(con, q)
        con.execute("DROP TABLE docs_fts")
        scan = R.bm25_scores(con, q)
        self.assertTrue(fts and scan, "оба пути пустые")
        top_fts = max(fts, key=lambda k: fts[k])
        top_scan = max(scan, key=lambda k: scan[k])
        self.assertEqual(top_fts, top_scan, "FTS5 находит другой top-hit, чем прежний проход")
        con.close()

    def test_device_reserve_survives_chunk_heavy_query(self):
        """Chunk-текст не должен вытеснять определения устройств из top-k."""
        con = self._mem(with_fts=True)
        out = R.search("либо мы либо они ложная дилемма выбора", top_k=2, min_score=0.05, con=con)
        self.assertTrue(any(r["kind"] == "device" for r in out),
                        "резерв устройств не сработал: top-k только из прозы")
        con.close()

    def test_single_meta_query(self):
        """Один запрос на всех кандидатов вместо SELECT на каждый документ.
        Служебные запросы FTS5 (docsize) не считаются — они неотделимы от bm25-ранжирования."""
        con = self._mem(with_fts=True)
        stmts = []
        def cb(*a):
            stmts.append(a[1] if len(a) == 3 else a[0])   # 3.12: (conn, sql, user); 3.14: (sql,)
        con.set_trace_callback(cb)
        R.search("ложная дилемма либо мы либо они", con=con)
        own = [s for s in stmts if "docs_fts_docsize" not in s and not s.startswith("--")]
        self.assertEqual([s for s in own if "WHERE id=?" in s], [],
                         f"per-doc SELECTs вернулись: {own}")
        self.assertLessEqual(len(own), 3, f"явных запросов больше, чем guard+match+meta: {own}")
        con.close()

    def test_query_connection_refuses_writes(self):
        """Запросная часть обязана открывать БД mode=ro — проверка поведением, не строкой."""
        import tempfile, sqlite3
        path = os.path.join(tempfile.mkdtemp(), "test.db")
        c0 = sqlite3.connect(path)
        c0.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, kind TEXT, label TEXT, text TEXT)")
        c0.commit(); c0.close()
        old = R.DB_PATH
        try:
            R.DB_PATH = path
            con = R.open_db()
            try:
                con.execute("CREATE TABLE hack(a INT)")
                self.fail("запросное соединение позволяет писать")
            except sqlite3.OperationalError:
                pass
            con.close()
        finally:
            R.DB_PATH = old

    def test_chunk_budget_applies_to_merged_output(self):
        """Бюджет MAX_OUT_CHARS — на слитый вывод, а не на каждый кусок: три куска не должны
        давать втрое больше обещанных 2500 символов."""
        con = self._mem(with_fts=True)
        con.executemany("INSERT INTO docs(kind, label, text) VALUES (?,?,?)",
                        [("chunk", f"Книга — Глава {i}", "дилемма " + "x" * 400) for i in range(20)])
        con.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
        chunks = ["дилемма " + "x" * 400] * 3
        out = R.merge_chunked(chunks, top_k=25, min_score=0.0, con=con)
        total = sum(len(r["snippet"]) + len(r["label"]) + 20 for r in out)
        self.assertLessEqual(total, R.MAX_OUT_CHARS,
                             f"слитый вывод больше обещанного бюджета: {total}")
        self.assertGreater(len(out), 1, "бюджет обязан ограничивать вывод, а не отключать его")
        con.close()


class WorkerTests(unittest.TestCase):
    """Тёплый worker: протокол JSON-lines, перезапуск после смерти процесса, деградация.
    Тесты идут без моделей (PSYCH_WORKER_NO_MODELS=1) — иначе каждый тест платит загрузку
    fastembed и ONNX."""

    def _db(self):
        import sqlite3
        path = os.path.join(tempfile.mkdtemp(), "test.db")
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, kind TEXT, label TEXT, text TEXT)")
        con.executemany("INSERT INTO docs(kind, label, text) VALUES (?,?,?)",
                        [("device", "Ложная дилемма", "только два варианта: либо A, либо B")])
        R.ensure_fts(con)
        con.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
        con.commit(); con.close()
        return path

    def _client(self):
        import worker_client as wc
        os.environ["PSYCH_WORKER_NO_MODELS"] = "1"
        os.environ["PSYCH_RAG_DB"] = self._db()
        return wc.WorkerClient(python=sys.executable)

    def test_protocol_answers_all_ops(self):
        c = self._client()
        hello = c.start()
        self.assertTrue(hello and hello.get("ok"), "worker не поздоровался")
        self.assertFalse(hello.get("embedding"), "в тестовом режиме модели грузятся")
        r = c.call("search", query="ложная дилемма")
        self.assertTrue(r and r.get("results"), "search через worker пустой")
        self.assertEqual(r["results"][0]["label"], "Ложная дилемма")
        r = c.call("markers", text="все знают, что этот приём работает")
        self.assertTrue(r and r.get("markers"), "markers через worker пустой")
        r = c.call("classify", text="абзац", min=0.70, gated=True)
        self.assertTrue(r and r.get("classification") == [], "classify без модели обязан вернуть пустой список")
        r = c.call("bogus")
        self.assertTrue(r and not r.get("ok") and r.get("error"), "неизвестный op молча проходит")
        c.stop()

    def test_client_restarts_after_process_death(self):
        import signal
        c = self._client()
        self.assertTrue(c.start())
        c.p.send_signal(signal.SIGKILL)
        r = c.call("ping")
        self.assertTrue(r and r.get("ok"), "клиент не перезапустил мёртвый worker")
        c.stop()

    def test_client_degrades_when_worker_unavailable(self):
        import worker_client as wc
        c = wc.WorkerClient(python="/nonexistent/python")
        self.assertIsNone(c.call("ping"), "недоступный worker должен возвращать None, а не падать")
        c.stop()


class CommentAnalysisTests(unittest.TestCase):
    """Единый входной модуль комментариев: один и тот же слой на всех трёх путях,
    отображение нейро-меток в корзины, приоритет тегов, общее определение треда."""

    ROWS = [{"id": "1", "parent": "root", "author": "a", "text": "отличная работа, спасибо"},
            {"id": "2", "parent": "1", "author": "b", "text": "злая бешеная критика без фактов"},
            {"id": "3", "parent": "1", "author": "c", "text": "купите сейчас, только сегодня скидка"}]

    SCORES = {"отличная работа, спасибо": {"emotion": {"joy": 0.9}, "toxicity": {"non-toxic": 0.9}, "sarcasm": {"positive": 0.8}},
              "злая бешеная критика без фактов": {"emotion": {"anger": 0.8}, "toxicity": {"insult": 0.75}, "sarcasm": {"neutral": 0.6}},
              "купите сейчас, только сегодня скидка": {"emotion": {"joy": 0.6}, "toxicity": {"non-toxic": 0.8}, "sarcasm": {"neutral": 0.7}}}

    def fake_runner(self):
        def run(texts, batch=32):
            return {name: [self.SCORES.get(t, {}).get(name, {}) for t in texts]
                    for name in ("emotion", "toxicity", "sarcasm")}
        return run

    def test_neural_bucket_mapping(self):
        import comment_analysis as CA
        res = CA.analyze(self.ROWS, runner=self.fake_runner())
        self.assertEqual(res["layer"], "neural")
        cats = [c["cat"] for c in res["comments"]]
        self.assertEqual(cats, ["радость", "злость", "радость"],
                         "нейро-метка не переведена в корзину доставляемого блока")
        self.assertEqual(res["mood"].get("злость"), 33)
        self.assertTrue(res["cross_check"], "кросс-проверка лексиконом пропала")

    def test_tag_priority(self):
        import comment_analysis as CA
        res = CA.analyze(self.ROWS, runner=self.fake_runner())
        tags = [c["tag"] for c in res["comments"]]
        self.assertEqual(tags[2], "💸", "продающий язык обязан быть приоритетнее эмоции")
        self.assertEqual(tags[1], "🤮", "токсичность p>=0.70 обязана давать 🤮")
        self.assertEqual(tags[0], "😊")

    def test_all_three_paths_use_the_same_entry_point(self):
        import comment_analysis as CA
        real = CA.neural_runner
        CA.neural_runner = lambda models=None: self.fake_runner()
        try:
            shared = CA.analyze(self.ROWS)
            if tele is not None:
                comments = [{"id": r["id"], "reply_to": None if r["parent"] == "root" else r["parent"],
                             "author": r["author"], "text": r["text"]} for r in self.ROWS]
                tg = tele.sentiment_block(comments)
                self.assertEqual(tg["mood"], shared["mood"],
                                 "Telegram-путь считает по другому слою")
                self.assertEqual(tg["homogeneity_overall"], shared["overall"])
                self.assertEqual(tg["layer"], "neural", "Telegram-путь остался на лексиконе")
            lex = CA.analyze(self.ROWS, use_neural=False)
            self.assertEqual(lex["layer"], "lexicon",
                             "без моделей слой обязан честно сообщить, что работает лексиконом")
            self.assertEqual(lex["mood"], lex["cross_check"])
        finally:
            CA.neural_runner = real

    def test_thread_definition_is_the_shared_one(self):
        import comment_analysis as CA
        res = CA.analyze(self.ROWS, runner=self.fake_runner())
        # reply-on-reply остаётся в том же треде: один тред с корнем id=1 и двумя потомками
        self.assertEqual(len(res["threads"]), 1)
        self.assertEqual(res["threads"][0]["size"], 3)
        # доминирует та категория, которой больше в треде (2 радость против 1 злость),
        # однородность = доля доминирующих из содержательных комментариев
        self.assertEqual(res["threads"][0]["dominant"], "радость")
        self.assertEqual(res["threads"][0]["homogeneity"], 67)
        self.assertEqual(res["overall"], 67)


if __name__ == "__main__":
    unittest.main(verbosity=2)
