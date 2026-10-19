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
import contextlib, importlib.util, io, json, os, re, sys, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
SKILL = os.path.dirname(SCRIPTS)
YT = os.path.join(SKILL, "youtube-content", "scripts")
PLUGIN = os.path.join(SKILL, "plugin", "psych-rag", "__init__.py")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, YT)

import markers, side
import ukr_detect as ukr, comment_sentiment, status_post, neural_pass
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
        p = load_plugin()
        src = read(PLUGIN)
        self.assertIn("--chunk", src, "плагин обязан опрашивать индекс кусками")
        ysrc = read(os.path.join(YT, "psych_layer.py"))
        self.assertIn("--chunk", ysrc, "psych_layer обязан опрашивать индекс кусками")
        self.assertNotIn("text[:1500]", ysrc, "psych_layer не должен резать запрос")

# --- B. правила, которые кодом не реализованы (падают) -----------------------
class TestOpenBugs(unittest.TestCase):
    def test_plugin_gate_equals_platform_limit(self):
        """Правило: гейт обязан равняться фактическому лимиту платформы (4096)."""
        p = load_plugin()
        src = read(PLUGIN)
        self.assertIn("4096", src, "в коде гейта 4000 — сообщения 4001–4096 не разбираются")

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
        src = read(os.path.join(YT, "status_post.py"))
        self.assertIn("4096", src, "нет контроля 4096 символов")
        self.assertIn("HTTPError", src, "нет обработки 429/сетевой ошибки")

    def test_tie_order_is_one_for_both_paths(self):
        """Один лексикон — один порядок ничьи. Проверка поведения, а не строки в файле:
        telegram-путь не держит свой словарь и обязан совпадать с comment_sentiment."""
        if tele is None:
            self.skipTest("telethon не установлен")
        src = read(os.path.join(YT, "telegram_comments.py"))
        self.assertIn("from comment_sentiment import", src,
                      "telegram-путь держит свой лексикон вместо общего")
        text = "злой и плохо"                      # ничья: злость и недовольство по одному
        lex = comment_sentiment.classify(text)[0]
        self.assertEqual(lex, "злость", "ничья в classify решается по TIE_ORDER")
        comments = [{"id": "R1", "reply_to": None, "text": text},
                    {"id": "R1.b1", "reply_to": "R1", "text": text}]
        block = tele.sentiment_block(comments)
        rows = [{"id": c["id"], "parent": c["reply_to"] or "root", "text": c["text"],
                 "author": None, "cat": comment_sentiment.classify(c["text"])[0]}
                for c in comments]
        th, by = comment_sentiment.build_threads(rows)
        lex_dom = comment_sentiment.thread_homogeneity(th, by)[0][0]["dominant"]
        self.assertEqual(block["threads"][0]["dominant"], lex_dom,
                         "ничья в треде на telegram-пути решается не по TIE_ORDER")

    def test_neural_labels_have_no_none_and_known_models(self):
        src = read(os.path.join(YT, "neural_pass.py"))
        self.assertIn("REPOS[name]", src)
        self.assertTrue(set(neural_pass.MULTILABEL) >= set(neural_pass.REPOS))
        self.assertIn("None not in labels", src, "нет защиты от None-метки в neural_pass")

    # --- новое, найденное этим проходом ---
    def test_runtime_classifier_operating_point(self):
        """Правило: ⚫ при p ≥ порога метки И p ≥ 0.70 — то есть max(порог, 0.70)."""
        plugin = read(PLUGIN)
        ysrc = read(os.path.join(YT, "psych_layer.py"))
        self.assertIn("--min", plugin, "плагин зовёт classifier без --min 0.70")
        self.assertIn("--min", ysrc, "psych_layer зовёт classifier без --min 0.70")
        self.assertIn("np.maximum(THRESH, min_conf)", read(os.path.join(SCRIPTS, "classifier.py")),
                      "операционная точка берётся как max(порог, min_conf)")

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
        block = tele.sentiment_block(comments)
        self.assertEqual(block["homogeneity_overall"], lex_hom,
                         f"telegram-путь даёт {block['homogeneity_overall']}% против {lex_hom}% "
                         "— вложенный ответ открывает отдельный тред")

    def test_overall_homogeneity_is_one_definition(self):
        src = read(os.path.join(YT, "rutube_comments.py"))
        self.assertNotIn("hom_sum / hom_n", src,
                         "rutube выдаёт как «в среднем по тредам» взвешенное среднее, "
                         "а навык определяет это как простое среднее по тредам")

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
        body = re.search(r"def lang_of.*?(?=\ndef )", read(os.path.join(YT, "ukr_detect.py")), re.S).group()
        self.assertIn("AMBIG", body, "AMBIG определён, но в решении не участвует: «це ми бачили де так» → ru")
        parts = body.split("return")[1:]
        self.assertTrue(any("weak_be" in p for p in parts),
                        "WEAK собирается, но никогда не влияет на вердикт")
        # поведение: неоднозначные без маркера — не русский; одна омонимичная форма язык не решает
        self.assertEqual(ukr.lang_of("це ми бачили де так"), "amb",
                         "неоднозначные слова без маркера уходили в ru")
        self.assertEqual(ukr.lang_of("па был в доме"), "ru",
                         "одна омонимичная форма не должна решать язык")

    def test_hf_language_filter_accepts_prefixed_tags(self):
        src = read(os.path.join(SCRIPTS, "model_update_check.py"))
        self.assertIn("language:", src, "фильтр языка ловит только 2-буквенные теги, `language:ru` пропускает")

    def test_fetch_transcript_outputs_promised_language_key(self):
        src = read(os.path.join(YT, "fetch_transcript.py"))
        self.assertIn('"language"', src.split("def main")[1],
                      "docstring обещает поле language в выводе, main его не печатает")

    def test_rag_mcp_chunks_long_query(self):
        src = read(os.path.join(SCRIPTS, "rag_mcp.py"))
        self.assertIn("chunk", src, "MCP-путь (основной по правилу навыка) не делит длинный запрос: "
                                    "embedding видит только первые ~512 токена")

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
        """Запросная часть читает БД в режиме ro (гейтвей держит write-лок) и не оставляет
        открытых дескрипторов."""
        src = read(os.path.join(SCRIPTS, "rag_core.py"))
        self.assertIn("mode=ro", src, "запросная часть открывает БД на запись")
        for f in sorted(os.listdir(SCRIPTS)):
            if f.endswith(".py") and f != "util.py":   # util.py — сам помощник, в его docstring шаблон упомянут намеренно
                s = read(os.path.join(SCRIPTS, f))
                self.assertNotIn("json.load(open(", s, f"{f}: незакрытый open()")

    def test_status_post_argv_guard(self):
        src = read(os.path.join(YT, "status_post.py"))
        self.assertIn("len(sys.argv)", src, "нет проверки аргументов — IndexError вместо usage")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
