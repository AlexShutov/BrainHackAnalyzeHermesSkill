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
import contextlib, importlib.util, io, json, os, re, subprocess, sys, tempfile, unittest
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
import classifier, rag_core as R, tag_routing as TR, util

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

try:
    import telegram_comments as tele
except Exception:
    tele = None

# Модули, которые тесты подменяют фиксами. Подменённый атрибут обязан быть восстановлен:
# `mod.os` — тот же объект, что у всего процесса, и невосстановленная подмена остаётся на
# следующие тесты (следующий тест проходит или падает не по своей логике — например пустой
# вывод вместо JSON). Снимок берётся по всем модулям навыка, загруженным в процесс (в том
# числе импортированным внутри теста), и восстанавливается в tearDown — «забыть finally»
# становится невозможным, а не зависимым от аккуратности конкретного теста.
def skill_modules():
    return {n: m for n, m in sys.modules.items()
            if (getattr(m, "__file__", "") or "").startswith(SCRIPTS)}


class SnapshotMixin:
    def setUp(self):
        self._snap = {n: dict(m.__dict__) for n, m in skill_modules().items()}

    def tearDown(self):
        for n, m in skill_modules().items():
            snap = self._snap.get(n)
            if not snap:
                continue
            for k, v in snap.items():
                setattr(m, k, v)

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
class TestFixed(SnapshotMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
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
        orig_client, orig_load = tele.client, tele.load_config
        tele.client = lambda: FakeClient()
        tele.load_config = lambda: {"api_id": 1, "api_hash": "x"}
        # патчить атрибут и восстанавливать в finally: tele.os — тот же объект os, что у
        # всего теста, и без восстановления подмена остаётся на следующие тесты
        # (проверка существования файла всегда даёт True). То же с tele.out: без
        # восстановления следующий тест печатает пустой вывод и падает на парсинге JSON,
        # а не на своей логике.
        orig_exists, orig_out = tele.os.path.exists, tele.out
        tele.os.path.exists = lambda p: True
        tele.out = lambda o: None
        try:
            class A: link, chat, post, limit, sentiment = "t.me/x/1", None, None, 0, False
            tele.cmd_comments(A())
            self.assertIsNone(seen.get("limit"), "0 должно превращаться в None («все»)")
            A.limit = 50
            tele.cmd_comments(A())
            self.assertEqual(seen.get("limit"), 50)
        finally:
            tele.os.path.exists = orig_exists
            tele.out, tele.client, tele.load_config = orig_out, orig_client, orig_load

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
class TestOpenBugs(SnapshotMixin, unittest.TestCase):
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
        orig = (status_post._urlopen, status_post.bot_token, status_post.time.sleep)
        status_post._urlopen = fake_urlopen
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

    def test_neural_pass_batches_and_has_one_loop_definition(self):
        """Поведение: правило — пачки по 32, поштучный forward на сотнях комментариев
        ~10× медленнее. Фикс считает вызовы forward: на 100 текстах при batch 32 обязано
        быть 4 вызова, а не 100. Пачка не имеет права менять метку: padding-хвосты не
        участвуют в attention (mask=0), поэтому метка текста в пачке обязана совпасть с
        меткой того же текста поштучно. И цикл батча обязан быть ОДИН (NP.infer), иначе
        правило выполняется только на одном пути."""
        calls = []

        def tok(texts, **kw):
            seqs = [[1] + [word_id(w) for w in re.findall(r"\w+", t)] + [2] for t in texts]
            L = max(len(s) for s in seqs)
            ids = [s + [0] * (L - len(s)) for s in seqs]
            mask = [[1] * len(s) + [0] * (L - len(s)) for s in seqs]
            return {"input_ids": ids, "attention_mask": mask}

        labels = ["neutral", "doubt"]

        def model(input_ids=None, attention_mask=None):
            calls.append(len(input_ids))
            rows = []
            for row_ids, row_mask in zip(input_ids, attention_mask):
                words = {ID2WORD.get(int(i), "") for i, m in zip(row_ids, row_mask) if m}
                logits = [-3.0, -3.0]
                if TARGET in words:
                    logits[1] = 3.0
                rows.append(logits)
            return rows

        texts = ["обычный текст"] * 100
        texts[50] = f"тут {TARGET} в середине"
        res = neural_pass.infer(tok, model, labels, texts, True, "emotion", batch=32)
        self.assertEqual(calls, [32, 32, 32, 4], f"forward вызван {len(calls)} раз: {calls}")
        self.assertEqual(res[50]["top"], "doubt", "пачка изменила метку текста")
        single = neural_pass.infer(tok, model, labels, texts, True, "emotion", batch=1)
        self.assertEqual([r["top"] for r in res], [r["top"] for r in single],
                         "батч-проход даёт другие метки, чем поштучный")

        # живой баг этого прохода: BatchEncoding — Mapping, а не dict, и путь падал
        # на enc.ids (AttributeError). Тест строит Mapping-объект, который НЕ dict.
        class MappingEnc:
            def __init__(self, data): self.data = data
            def __getitem__(self, k): return self.data[k]
        enc = MappingEnc({"input_ids": [[1, 2], [1, 0]], "attention_mask": [[1, 1], [1, 0]]})
        self.assertNotIsInstance(enc, dict, "тест обязан ловить именно Mapping, а не dict")
        ids, mask = neural_pass._enc_pair(enc)
        self.assertEqual(list(ids[1]), [1, 0], "Mapping-путь потерял строку")
        self.assertEqual(list(mask[1]), [1, 0], "padding-маска потерялась")

        src = read(os.path.join(YT, "comment_analysis.py"))
        self.assertIn("NP.infer(", src,
                      "comment_analysis держит свой цикл батча — два источника одного правила")

    def test_fetch_comments_caches_by_video_id_and_carries_id(self):
        """Поведение: правило — повторный анализ того же видео не перекачивает (кэш по
        video id в scratch), и каноническая форма комментария обязана несть `id`: без него
        build_threads не строит тред по цепочке, а угадывает корень по префиксу parent.
        Тест без сети: runner подменён, кэш — временный каталог.
        Partial-набор (yt-dlp сам остановился) обязан перекачиваться, когда запрошено
        больше, чем лежит: выдавать partial за полный нельзя. И скрипт не имеет права
        выдумывать опцию — --max-comments в этой версии yt-dlp нет (проверено на --help),
        поэтому --limit обрезает только вывод, а полнота проверяется размером кэша."""
        import fetch_comments as FC
        cache = tempfile.mkdtemp()
        calls = []

        def runner(url):
            calls.append(url)
            return {"title": "T", "webpage_url": url,
                    "comments": [{"id": f"R{i}", "parent": None, "author": "a",
                                  "text": f"текст {i}", "like_count": i}
                                 for i in range(3)]}

        url = "https://www.youtube.com/watch?v=aaaaaaaaaaa&t=123"
        self.assertEqual(FC.video_id(url), "aaaaaaaaaaa",
                         "запрос ?t=123 обязан обрезаться до id")
        self.assertNotIn("--max-comments", FC.build_cmd(url),
                         "выдуманная опция yt-dlp: качание комментариев не ограничивается")

        meta, comments, hit = FC.fetch(url, limit=10, runner=runner, cache_dir=cache)
        self.assertEqual(len(calls), 1, "первый заход обязан качать")
        self.assertFalse(hit, "первый заход не может быть попаданием в кэш")
        self.assertTrue(all(c["id"] for c in comments), "каноническая форма без поля id")

        meta2, comments2, hit2 = FC.fetch(url, limit=2, runner=runner, cache_dir=cache)
        self.assertTrue(hit2, "кэш покрывает запрос — обязан сработать без сети")
        self.assertEqual(len(calls), 1, "покрытие кэша не имеет права вызывать runner")
        self.assertEqual(len(comments2), 2, "кэш обязан уважать запрошенный limit")

        meta3, comments3, hit3 = FC.fetch(url, limit=10, runner=runner, cache_dir=cache)
        self.assertFalse(hit3, "partial-кэш (3 из запрошенных 10) нельзя выдавать за полный")
        self.assertEqual(len(calls), 2, "partial-кэш обязан вызвать перекачивание")

    def test_fetch_comments_finds_yt_dlp_without_manual_pythonpath(self):
        """Поведение: yt-dlp живёт в изолированном target dir, а не в venv навыка.
        Замер на этой машине: find_spec('yt_dlp') -> None, и subprocess падал с
        ModuleNotFoundError, пока PYTHONPATH не задали вручную. Скрипт обязан сам
        добавить fallback-каталог в env ребёнка и не затирать уже заданный."""
        import fetch_comments as FC
        import importlib.util as u
        if u.find_spec("yt_dlp") is not None:
            self.skipTest("yt_dlp импортируется в этом interpreters — fallback не нужен")
        if not os.path.isdir(FC.FALLBACK_PYLIB):
            self.skipTest(f"fallback-каталог {FC.FALLBACK_PYLIB} не установлен")
        saved = os.environ.get("PYTHONPATH")
        try:
            os.environ.pop("PYTHONPATH", None)
            env = FC.child_env()
            self.assertIn("yt-pylib", env.get("PYTHONPATH", ""),
                          "fallback-каталог не добавлен — скрипт падает на ModuleNotFoundError")
            os.environ["PYTHONPATH"] = "/x"
            env2 = FC.child_env()
            self.assertTrue(env2.get("PYTHONPATH", "").endswith("/x"),
                            "уже заданный PYTHONPATH обязан сохраниться")
        finally:
            if saved is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = saved

    def test_promote_gate_refuses_foreign_thresholds(self):
        """Поведение: пороги — свойство конкретной модели. Тест строит staging, где пороги
        кандидата байтово совпадают с живыми (это пороги другой модели), и где их число не
        совпадает с числом меток; в обоих случаях продвижение отказывает и не трогает живую
        модель. Отдельно — отказ по неготовому состоянию и по отсутствию порогов, и --force
        как единственный обход."""
        import model_promote as MP
        d = tempfile.mkdtemp()
        models, staging = os.path.join(d, "models"), os.path.join(d, "staging")
        os.makedirs(models); os.makedirs(staging)
        open(os.path.join(models, "classifier_int8.onnx"), "wb").write(b"LIVE")
        open(os.path.join(staging, "classifier_int8.onnx"), "wb").write(b"NEW")
        import numpy as np
        np.save(os.path.join(models, "thresholds.npy"), np.array([0.7, 0.6]))
        state = os.path.join(models, "update_state.json")
        orig_models, orig_state = MP.MODELS, MP.STATE
        MP.MODELS, MP.STATE = models, state
        try:
            def write_state(status):
                with open(state, "w", encoding="utf-8") as f:
                    json.dump({"status": status, "staging": staging,
                               "candidate": {"model_id": "x"}}, f)
            def run():
                buf = io.StringIO()
                argv = sys.argv
                sys.argv = ["model_promote.py"]
                try:
                    with contextlib.redirect_stderr(buf):
                        rc = MP.main()
                finally:
                    sys.argv = argv
                return rc, buf.getvalue()

            write_state("needs_update")
            rc, err = run()
            self.assertEqual(rc, 1, "продвижение неготового состояния не отказало")
            self.assertIn("нужна успешная проверка", err)

            write_state("ready")
            np.save(os.path.join(staging, "thresholds.npy"), np.array([0.7, 0.6]))
            rc, err = run()
            self.assertEqual(rc, 1, "пороги другой модели продвинулись молча")
            self.assertIn("идентичны", err)
            self.assertEqual(open(os.path.join(models, "classifier_int8.onnx"), "rb").read(), b"LIVE",
                             "отказ обязан не трогать живую модель")

            np.save(os.path.join(staging, "thresholds.npy"), np.array([0.5]))
            with open(os.path.join(staging, "labels.txt"), "w", encoding="utf-8") as f:
                f.write("a\nb\n")
            rc, err = run()
            self.assertEqual(rc, 1, "несовпадение длины порогов и меток не отказало")
            self.assertIn("не совпадает", err)

            os.remove(os.path.join(staging, "thresholds.npy"))
            rc, err = run()
            self.assertEqual(rc, 1, "кандидат без своих порогов продвинулся молча")
            self.assertIn("калибровка", err)

            np.save(os.path.join(staging, "thresholds.npy"), np.array([0.5, 0.4]))
            rc, out = run()
            self.assertEqual(rc, 0, f"корректное продвижение отказало: {out}")
            self.assertEqual(open(os.path.join(models, "classifier_int8.onnx"), "rb").read(), b"NEW")
        finally:
            MP.MODELS, MP.STATE = orig_models, orig_state

    def test_version_check_is_once_per_chat_and_once_per_day(self):
        """Поведение: ключ проверки версии — (platform, chat_id), а не процесс: новый чат
        обязан получить свою проверку, прежний — не раньше суток. Тест подменяет subprocess
        и время (и восстанавливает их — без восстановления подмена остаётся на весь прогон)
        и считает вызовы."""
        import types
        pl = load_plugin()
        calls = []

        def fake_run(args, **kw):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout='{"status": "needed"}', stderr="")

        class FakeSrc:
            def __init__(self, p, c): self.platform, self.chat_id = p, c

        orig_run, orig_time = pl.subprocess, pl.time
        pl.subprocess, pl.time = types.SimpleNamespace(run=fake_run), types.SimpleNamespace(time=lambda: 1000.0)
        try:
            pl._checked.clear()
            self.assertIsNotNone(pl.version_check(FakeSrc("telegram", 111)),
                                 "первое сообщение чата обязано дать проверку")
            self.assertEqual(len(calls), 1)
            self.assertIsNone(pl.version_check(FakeSrc("telegram", 111)),
                              "тот же чат не имеет права перепроверять в пределах суток")
            self.assertEqual(len(calls), 1, "лишний subprocess на тот же чат в пределах суток")
            self.assertIsNotNone(pl.version_check(FakeSrc("telegram", 222)),
                                 "новый чат обязан получить свою проверку")
            self.assertEqual(len(calls), 2)
            pl.time = types.SimpleNamespace(time=lambda: 1000.0 + pl.RECHECK_SECONDS + 1)
            self.assertIsNotNone(pl.version_check(FakeSrc("telegram", 111)),
                                 "сутки прошли — чат обязан быть перепроверен")
            self.assertEqual(len(calls), 3)
        finally:
            pl.subprocess, pl.time = orig_run, orig_time
            pl._checked.clear()

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

    def test_cli_default_applies_operating_point(self):
        """Правило: ⚫ при p ≥ порога метки И p ≥ 0.70. CLI без --min обязан применять
        рабочую точку, а не только thresholds.npy — кейс, где «как есть» и «по правилу»
        дают разный результат (Doubt p=0.60, её порог 0.47)."""
        class FixedSession:
            def __init__(self, probs): self.probs = probs
            def run(self, _, feeds):
                logits = np.full(len(classifier.LABELS), -3.0)
                for name, p in self.probs.items():
                    logits[classifier.LABELS.index(name)] = np.log(p / (1 - p))
                return [np.array([logits])]
        classifier.session = lambda: FixedSession({"Doubt": 0.60, "Appeal_to_Time": 0.80})
        classifier.tokenizer = lambda: FakeTok()
        labels = {l["label"] for r in run_classifier("абзац")["classification"] for l in r["labels"]}
        self.assertNotIn("Doubt", labels,
                        "CLI без --min применяет только порог 0.47 — рабочая точка не применяется")
        self.assertIn("Appeal_to_Time", labels)
        # рабочая точка — default, а не жёсткий пол: явный --min вниз обязан её снять
        labels2 = {l["label"] for r in run_classifier("абзац", ["--min", "0.5"])["classification"]
                   for l in r["labels"]}
        self.assertIn("Doubt", labels2, "--min 0.5 обязан вернуть p=0.60")
        # одна точка во всех путях: CLI, worker, плагин, psych_layer
        import psych_layer as PL
        p = load_plugin()
        self.assertEqual(p.OPERATING_POINT, classifier.OPERATING_POINT,
                        "рабочая точка разошлась между плагином и CLI")
        self.assertEqual(PL.OPERATING_POINT, classifier.OPERATING_POINT,
                        "рабочая точка разошлась между psych_layer и CLI")
        self.assertEqual(classifier.OPERATING_POINT, util.OPERATING_POINT,
                        "CLI не читает рабочую точку из общего модуля")
        self.assertEqual(p.OPERATING_POINT, util.OPERATING_POINT,
                        "плагин не читает рабочую точку из общего модуля")

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

    def test_skill_dictionaries_have_no_latin_homoglyphs(self):
        """Поведение: homoglyph внутри кириллического слова не даёт синтаксической ошибки —
        слово просто никогда не совпадёт, и правило молча перестаёт работать. Тест вставляет
        подмену и требует, чтобы сканер её нашёл: без этого проверка вырождается в «сейчас чисто».
        """
        def scan(text):
            return [w for w in re.findall(r"[A-Za-zА-Яа-яё]+", text)
                    if re.search(r"[а-яё]", w) and re.search(r"[A-Za-z]", w)]
        planted = "злостnая"          # кириллическое слово с латинской n внутри
        self.assertEqual(scan(planted), [planted], "сканер не ловит подмену — проверка бессмысленна")
        self.assertEqual(scan("ok"), [], "латинское слово само по себе не подмена")

        def words(obj, path=""):
            """Словарные списки — вложенные dict/list: обходить рекурсивно и проверять
            СЛОВО целиком, а не строку: «Levinson» в скобках — латинское имя, а не подмена."""
            out = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    out.extend(words(k, path) + words(v, path))
            elif isinstance(obj, (list, tuple, set)):
                for x in obj:
                    out.extend(words(x, path))
            else:
                for w in re.findall(r"[A-Za-zА-Яа-яё]+", str(obj)):
                    if re.search(r"[а-яё]", w) and re.search(r"[A-Za-z]", w):
                        out.append((path, w))
            return out

        planted = "злостnая"          # кириллическое слово с латинской n внутри
        self.assertEqual(words(planted, "planted"), [("planted", planted)],
                         "сканер не ловит подмену — проверка бессмысленна")
        self.assertEqual(words("Levinson 1983", "name"), [],
                         "латинское имя в скобках — не подмена")

        sources = {"comment_sentiment": (comment_sentiment.ANGER, comment_sentiment.JOY,
                                         comment_sentiment.DISCONTENT, comment_sentiment.NEG,
                                         comment_sentiment.CATS),
                   "ukr_detect": (ukr.UK_WORDS, ukr.BE_WORDS, ukr.AMBIG, ukr.WEAK,
                                  ukr.SHARED_UK_BE, ukr.UK_LETTERS, ukr.BE_ONLY),
                   "markers": (markers.MARKERS, markers.STOP, markers.MAP,
                               markers.FAMILIES, markers.NAMES)}
        bad = []
        for name, objs in sources.items():
            for obj in objs:
                bad.extend(words(obj, name))
        self.assertEqual(bad, [], "словарный список содержит латинскую букву внутри кириллического слова: "
                                  + "; ".join(f"{p}: {w}" for p, w in bad))

    def test_homoglyph_in_dictionary_breaks_routing_and_is_caught(self):
        """Поведение: homoglyph в словаре — не синтаксическая ошибка, а молчаливый пропуск
        правила. Семья «номинализация и удаление агента» была написана с латинской n
        («номinalizация»), и `NLP_FAMILY_MARKS` («номинализаци») её не видела: hit в этой
        семье не давал подтипа вообще — None вместо 🔪🧠. Тест берёт имя семьи из самого
        markers.json и прогоняет его через routing, а не пишет имя сам."""
        p = os.path.join(SKILL, "psychology", "markers.json")
        with open(p, encoding="utf-8") as f:
            fams = json.load(f)["families"]
        fam = [k for k in fams if "номинализ" in k.lower()]
        self.assertTrue(fam, "в markers.json нет семьи номинализации — тест не о чём")
        self.assertEqual(TR.pair_for_family(fam[0]), "🔪🧠",
                         f"имя семьи {fam[0]!r} не совпадает с NLP_FAMILY_MARKS — подтип теряется")
        self.assertEqual(markers.unmapped_families(), [], "MAP и markers.json разошлись")

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

    def test_status_post_not_modified_guard_is_reachable(self):
        """Правило: guard «message is not modified» обязан быть достижим. Раньше api()
        выходил на любом не-429 HTTPError, и heartbeat с той же строкой статуса ронял
        скрипт, а пост зависал на последом успешном обновлении. Кейс, где «как есть»
        (падение) и «по правилу» (guard) дают разный результат. Сетевой скрипт тестируется
        через dry-run + mock-transport, а не через чтение исходника."""
        import urllib.error, urllib.parse
        calls = []

        def http_err(desc):
            return urllib.error.HTTPError("u", 400, "Bad Request", {},
                                          io.BytesIO(json.dumps({"ok": False,
                                                                 "description": desc}).encode()))

        def fake(req, timeout=15):
            calls.append(req)
            text = urllib.parse.parse_qs(req.data.decode()).get("text", [""])[0]
            if "[SAME]" in text:
                raise http_err("Bad Request: message is not modified")
            if "[TOOLONG]" in text:
                raise http_err("Bad Request: message is too long")
            class Resp:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self): return json.dumps({"ok": True, "result": {"message_id": 5}}).encode()
            return Resp()

        def run(argv_list, stream):
            argv, sys.argv = sys.argv, argv_list
            buf = io.StringIO()
            ctx = contextlib.redirect_stdout(buf) if stream == "out" else contextlib.redirect_stderr(buf)
            try:
                with ctx:
                    status_post.main()
            except SystemExit as e:
                buf.write(str(e))          # выход скрипта — тоже результат теста
            finally:
                sys.argv = argv
            return buf.getvalue()

        orig = (status_post._urlopen, status_post.bot_token, status_post.STATE_DIR)
        status_post._urlopen, status_post.bot_token = fake, lambda: "T"
        status_post.STATE_DIR = tempfile.mkdtemp()
        try:
            with open(os.path.join(status_post.STATE_DIR, "k.json"), "w") as f:
                json.dump({"chat_id": "123", "message_id": 5, "preview": "превью"}, f)

            # A: heartbeat с той же строкой — guard обязан сработать, скрипт не падает
            out = run(["status_post.py", "update", "k", "статус [SAME]"], "out")
            self.assertIn("updated", out,
                          "guard недостижим: api() выходит на не-429 HTTPError, "
                          "heartbeat с той же строкой роняет скрипт")

            # B: реальная ошибка не проглатывается тем же guard-ом
            err = run(["status_post.py", "update", "k", "статус [TOOLONG]"], "err")
            self.assertIn("HTTP 400", err, "настоящая ошибка обязана остаться ошибкой")

            # C: dry-run печатает, что отправил бы, и не пишет ничего
            calls.clear()
            out3 = run(["status_post.py", "start", "123", "drykey", "превью", "статус", "--dry-run"], "out")
            self.assertIn("dry-run", out3, "dry-run обязан печатать, что отправил бы")
            self.assertIn("превью", out3, "превью обязано сохраниться в dry-run-выводе")
            self.assertEqual(calls, [], "dry-run обязан не делать сетевых запросов")
            self.assertFalse(os.path.exists(os.path.join(status_post.STATE_DIR, "drykey.json")),
                             "dry-run обязан ничего не писать")
        finally:
            (status_post._urlopen, status_post.bot_token, status_post.STATE_DIR) = orig

    def test_telegram_comments_is_testable_without_network(self):
        """Поведение: сетевой скрипт становится тестируемым только когда у него есть
        offline/dry-run режим и mock-transport. Здесь dry-run печатает план и НЕ создаёт
        клиент, и обязан работать БЕЗ сессии: прежний порядок проверял логин ДО dry-run,
        и offline-режим был недоступен именно там, где нужен. Mock-клиент передаётся
        извне и даёт каноническую форму комментария."""
        if tele is None:
            self.skipTest("telethon не установлен — путь не импортируется")
        calls = []

        def boom():
            calls.append("client")
            raise AssertionError("dry-run обязан не создавать клиент")

        orig_client, orig_load, orig_session = tele.client, tele.load_config, tele.SESSION
        tele.client, tele.load_config, tele.SESSION = boom, lambda: None, "/nonexistent/session"
        try:
            a = types.SimpleNamespace(link="https://t.me/c/1543/100", chat=None, post=None,
                                      limit=0, sentiment=False, dry_run=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tele.cmd_comments(a)
            res = json.loads(buf.getvalue())
            self.assertEqual(res["status"], "dry_run", "dry-run не работает")
            self.assertEqual(calls, [], "dry-run создал клиент")
            self.assertIsNone(res["limit"], "limit=0 обязан означать «все» (None)")
            self.assertEqual(res["post"], 100, "id поста ушёл в query (?t=123)")

            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                tele.cmd_comments(types.SimpleNamespace(
                    link="https://t.me/c/1543/100", chat=None, post=None,
                    limit=0, sentiment=False, dry_run=False))
            self.assertEqual(json.loads(buf2.getvalue())["status"], "login_required",
                             "проверка сессии пропала")
        finally:
            tele.client, tele.load_config, tele.SESSION = orig_client, orig_load, orig_session

        class FakeMsg:
            def __init__(self, i, text, reply=None):
                self.id, self.text = i, text
                self.reply_to = types.SimpleNamespace(reply_to_msg_id=reply) if reply else None
                self.sender = types.SimpleNamespace(first_name="A", sender_id="A")
                self.post_author, self.date, self.reactions = None, None, None

        seen = []

        class FakeClient:
            def get_entity(self, peer):
                seen.append(("entity", peer))
                return "e"
            def iter_messages(self, entity, reply_to=None, limit=None):
                seen.append(("iter", limit))
                return iter([FakeMsg(1, "злой текст"), FakeMsg(2, "злой текст", reply=1)])

        rows = tele.fetch_comments(FakeClient(), "-1001543", 100, 0)
        self.assertEqual(seen[1][1], None, "limit=0 обязан означать «все» (None)")
        self.assertTrue(all(r["id"] for r in rows), "каноническая форма без поля id")
        self.assertEqual(rows[1]["reply_to"], 1, "reply_to потерялся")

    def test_snapshot_mixin_restores_patched_module_attribute(self):
        """Поведение: подмена атрибута модуля обязана восстанавливаться после теста.
        Замер contamination-класса: `tele.out = lambda o: None` без finally дал следующему
        тесту пустой вывод и падение на json.loads — тест проходил отдельно и падал в полном
        прогоне, то есть ровно тот ложеловный «успех», о котором правило."""
        class T(SnapshotMixin, unittest.TestCase):
            def runTest(self):
                pass
        t = T("runTest")
        t.setUp()
        orig = markers.scan
        markers.scan = lambda *a, **k: []
        self.assertIsNot(markers.scan, orig)
        t.tearDown()
        self.assertIs(markers.scan, orig)

    def test_all_test_classes_use_the_snapshot_mixin(self):
        """Guard-проверка (не поведенческая): каждый тестовый класс обязан наследовать
        SnapshotMixin — иначе подмена остаётся на следующие тесты."""
        src = read(os.path.join(SCRIPTS, "tests", "test_layers.py"))
        classes = re.findall(r"^class (\w+)\(([^)]*)\):", src, re.M)
        bad = [c for c, bases in classes if c != "SnapshotMixin" and "SnapshotMixin" not in bases]
        self.assertEqual(bad, [], "классы без снимка: " + ", ".join(bad))

class TagRuleTests(SnapshotMixin, unittest.TestCase):
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

    def test_block_states_unavailable_neural_layer(self):
        rows = [{"id": "R1", "author": "a", "text": "злой и злой", "parent": "root", "likes": 0}]
        res = CA.analyze(rows, runner=lambda texts: None, use_neural=True, use_markers=False)
        self.assertTrue(res["neural_unavailable"])
        self.assertIn("недоступен", CA.block(res), "лексикон подан как нейро-вердикт")
        # и не врёт, когда нейро не запрашивали: там лексикон по явному выбору для сверки
        res2 = CA.analyze(rows, use_neural=False, use_markers=False)
        self.assertFalse(res2["neural_unavailable"])
        self.assertNotIn("недоступен", CA.block(res2), "лишняя оговорка, когда нейро не запрашивали")

    def test_device_tag_is_pair_not_name(self):
        self.assertEqual(CA.tag("текст", "нейтрально", {}, "False_Dilemma-No_Choice"), "🎭🧠")
        self.assertEqual(CA.tag("текст", "нейтрально", {}, "Slogans"), "📢🧠")
        self.assertEqual(CA.tag("текст", "нейтрально", {}, "🔪🧠"), "🔪🧠")
        # имя устройства без пары — незавершённая метка: подтип решается по сработавшей семье
        self.assertEqual(TR.pair_for_device("Пресуппозиция",
                                            "пресуппозиционные триггеры (Levinson 1983)"), "🔪🧠")
        self.assertIsNone(TR.pair_for_device("Пресуппозиция"),
                          "без сработавшей семьи и без вердикта пары нет — метка неполная")

    def test_comment_block_can_actually_show_device_tag(self):
        # раньше device_by_text не передавал ни один из трёх путей: «приём» был невозможен
        ms, cs = markers.scan, classifier.classify
        markers.scan = lambda txt: [{"family": "структурные маркеры", "marker": "repetition",
                                     "para": 1, "found": "x", "context": "y",
                                     "candidates": ["Ложная дилемма"]}]
        classifier.classify = lambda text, min_conf=None: [
            {"label": "False_Dilemma-No_Choice", "p": 0.80, "threshold": 0.57, "chunk": 1,
             "ru": "Ложная дилемма", "device": "Ложная дилемма"}]
        try:
            rows = [{"id": "R1", "author": "a", "text": "либо A, либо B", "parent": "root", "likes": 0}]
            res = CA.analyze(rows, use_neural=False)
            self.assertEqual(res["comments"][0]["tag"], "🎭🧠",
                             "блок комментариев не может показать приём")
            self.assertIn("🎭🧠", CA.block(res))
        finally:
            markers.scan, classifier.classify = ms, cs

    def test_classifier_verdict_attaches_to_comment_paragraph(self):
        # classify() не несёт `para` (её добавляет worker): без атрибуции вердикт модели
        # теряется, и комментарий с ложной дилеммой остаётся без пары
        ms, cs = markers.scan, classifier.classify
        markers.scan = lambda txt: [{"family": "структурные маркеры", "marker": "repetition",
                                     "para": 1, "found": "x", "context": "y",
                                     "candidates": ["Ложная дилемма"]}]
        classifier.classify = lambda text, min_conf=None: [
            {"label": "False_Dilemma-No_Choice", "p": 0.80, "threshold": 0.57, "chunk": 1,
             "ru": "Ложная дилемма", "device": "Ложная дилемма"}]
        try:
            self.assertEqual(CA.device_pairs("либо A, либо B"), "🎭🧠")
            # гейт: без маркеров классификатор не вызывается и пара не ставится
            markers.scan = lambda txt: []
            self.assertIsNone(CA.device_pairs("либо A, либо B"))
        finally:
            markers.scan, classifier.classify = ms, cs

    def test_dilts_families_route_to_nlp(self):
        import psych_layer as PL
        hit = {"family": "Фокусы языка: намерение и переопределение", "marker": "m",
               "para": 1, "found": "x", "context": "y"}
        out = PL.attach_subtypes([{"para": 1, "text": "абзац", "labels": []}], [hit])[0]
        self.assertEqual(out["subtype"], "🔪🧠", "фокус Дилтса оставался без подтипа")

    def test_one_routing_dictionary_for_both_paths(self):
        import psych_layer as PL
        self.assertIs(PL.LABEL_SUBTYPE, TR.LABEL_SUBTYPE)
        self.assertIs(PL.NLP_FAMILY_MARKS, TR.NLP_FAMILY_MARKS)

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

    def test_marketing_requires_the_device_not_a_similar_word(self):
        """💸 — за сам продающий приём, а не за слово, которое похоже на продажу.
        Прежний плоский список (подстрока) ловил бытовые фразы: «осталось только
        подождать», «это для вас», «90% комментариев», «источник по ссылке» — ни в
        одной из них нет приёма. Каждый настоящий приём обязан сработать по группе
        (триггер + окружение), и block() обязан назвать приём, а не только модификатор."""
        for text in ["осталось только подождать", "это для вас",
                     "90% комментариев", "источник по ссылке"]:
            self.assertIsNone(CA.marketing_device(text),
                              f"бытовая фраза без приёма даёт 💸: {text}")
            self.assertNotEqual(CA.tag(text, "нейтрально", {}), "💸",
                                f"тег поставлен без приёма: {text}")
        cases = [("купите сейчас", "призыв к действию"),
                 ("осталось мало штук, только сегодня", "дефицит/срочность"),
                 ("обычная цена 1000, для вас 700", "ценовое якорение"),
                 ("выбор 90% покупателей",
                  "социальное доказательство с непроверяемой цифрой"),
                 ("учёные доказали, что это работает", "лженаучная апелляция"),
                 ("подарок за первую покупку", "подарок за немедленную покупку"),
                 ("переход по ссылке со скидкой", "ссылка как канал продажи")]
        for text, name in cases:
            self.assertEqual(CA.marketing_device(text), name,
                             f"приём {name} не назван по тексту: {text}")
            self.assertEqual(CA.tag(text, "нейтрально", {}), "💸")
        # доставляемый блок называет конкретный приём, а не только модификатор
        res = CA.analyze([{"id": "1", "parent": "root", "author": "a",
                           "text": "обычная цена 1000, для вас 700", "likes": 0}],
                          use_neural=False, use_markers=False)
        out = CA.block(res)
        self.assertIn("ценовое якорение", out, "блок не называет приём")
        self.assertIn("💸", out)


class RagPathTests(SnapshotMixin, unittest.TestCase):
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


class WorkerTests(SnapshotMixin, unittest.TestCase):
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
        r = c.call("neural", texts=["текст"])
        self.assertTrue(r and not r.get("ok") and r.get("error") == "no models",
                        "op `neural` в протоколе отсутствует: worker отвечает unknown op, "
                        "и путь комментариев не может ходить через тёплый worker")
        c.stop()

    def test_client_defaults_to_the_interpreter_that_has_the_models(self):
        """guard-проверка: worker обязан запускаться интерпретатором, где модели есть.
        Путь комментариев может идти под системным python3 (на этой машине — 3.9 без
        transformers), и клиент брал sys.executable: load_neural падал, worker отвечал
        «neural unavailable», и анализ молча деградирует к лексикону — читатель видел
        числа словаря, не зная, что warm-путь был. Ограничение: тесты идут под venv,
        поэтому там venv и sys.executable совпадают, и разница видна только в ветке env;
        живой proof — прогон под системным python3 (он и поймал баг)."""
        import worker_client as wc
        old = os.environ.get("PSYCH_RAG_PY")
        try:
            os.environ["PSYCH_RAG_PY"] = "/bin/false-python"
            self.assertEqual(wc.resolve_python(), "/bin/false-python",
                             "env обязан перебивать venv")
            os.environ.pop("PSYCH_RAG_PY")
            if not os.path.exists(wc.VENV_PY):
                self.skipTest("venv с моделями не установлен — проверка не о чём")
            self.assertEqual(wc.resolve_python(), wc.VENV_PY,
                             "без env клиент обязан брать venv, а не sys.executable")
        finally:
            if old is not None:
                os.environ["PSYCH_RAG_PY"] = old

    def test_worker_sees_neural_pass_from_the_comment_path(self):
        """guard-проверка пути импорта: neural_pass живёт в youtube-content/scripts, а
        worker — в scripts/. Без правильного пути импорт падает, worker отвечает «neural
        unavailable», и путь комментариев молча возвращается к загрузке моделей в каждом
        процессе — ровно тот баг, который тест ловит. Живой замер (2026-10-19): первый
        запрос через worker 10.8 с (ленивая загрузка трёх моделей), каждый следующий
        0.04 с; прежний путь платил загрузку на каждый анализ."""
        import worker as W
        p = W.neural_paths()
        target = os.path.join(p, "neural_pass.py")
        self.assertTrue(os.path.exists(target),
                        "путь worker не указывает на модуль нейро-прохода")
        sys.path.insert(0, p)
        try:
            import neural_pass as NP
            self.assertEqual(os.path.abspath(NP.__file__), target,
                             "worker импортирует другой модуль")
        finally:
            sys.path.remove(p)
            sys.modules.pop("neural_pass", None)

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


class CommentAnalysisTests(SnapshotMixin, unittest.TestCase):
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

    def test_curiosity_has_a_bucket_and_mapping_is_total(self):
        """Правило: нейро-метка переводится в корзину доставляемого блока. Было 27 из 28:
        `curiosity` без корзины — комментарий молча уходил в лексикон, и доставляемые
        цифры смешивали два слоя, не говоря об этом."""
        missing = [l for l in neural_pass.LABELS["emotion"] if l not in CA.NEURAL_BUCKET]
        self.assertEqual(missing, [], f"метки модели без корзины: {missing}")

        # кейс, где «как есть» (лексикон) и «по правилу» (нейро) дают РАЗНЫЙ результат
        text = "как он это сделал — интересно, но плохо"
        lex = comment_sentiment.classify(text)[0]
        self.assertNotEqual(lex, "нейтрально",
                            "тест бессмысленен: оба слоя дают одно и то же")
        def runner(texts):
            return {"emotion": [{"curiosity": 0.9} for _ in texts],
                    "toxicity": [{"non-toxic": 0.9} for _ in texts],
                    "sarcasm": [{"neutral": 0.9} for _ in texts]}
        res = CA.analyze([{"id": "R1", "author": "a", "text": text, "parent": "root", "likes": 0}],
                         runner=runner, use_markers=False)
        self.assertEqual(res["comments"][0]["cat"], "нейтрально",
                         "curiosity без корзины — комментарий ушёл в лексикон")
        self.assertEqual(res["unmapped"], [], "метка обязана быть покрыта корзиной")

        # метка вне списка (пропуск в id2label) — блок обязан сказать, что число получил словарь
        def runner2(texts):
            return {"emotion": [{"class_99": 0.9} for _ in texts],
                    "toxicity": [{"non-toxic": 0.9} for _ in texts],
                    "sarcasm": [{"neutral": 0.9} for _ in texts]}
        res2 = CA.analyze([{"id": "R1", "author": "a", "text": text, "parent": "root", "likes": 0}],
                           runner=runner2, use_markers=False)
        out = CA.block(res2)
        self.assertIn("получена лексиконом", out, "блок молчит о смешивании слоёв")
        self.assertIn("class_99", out, "блоку нечего назвать")

    def test_tag_priority(self):
        import comment_analysis as CA
        res = CA.analyze(self.ROWS, runner=self.fake_runner())
        tags = [c["tag"] for c in res["comments"]]
        self.assertEqual(tags[2], "💸", "продающий язык обязан быть приоритетнее эмоции")
        self.assertEqual(tags[1], "🤮", "токсичность p>=0.70 обязана давать 🤮")
        self.assertEqual(tags[0], "😊")

    def test_mood_percentages_sum_to_100(self):
        """Доставляемые числа — hard assertion: распределение обязано складываться
        в 100. Прежний `round()` по каждой категории на трёх равных категориях давал
        33+33+33=99 — блок печатал распределение, которое противоречит само себе.
        Проверка поведением: три комментария в трёх категориях, сумма процентов в
        доставляемом блоке = 100; тот же кейс на прежнем round() обязан дать 99 —
        иначе тест не доказывает разный результат. Кросс-проверка лексиконом (слой
        neural) обязана считать тем же методом."""
        import re
        import comment_analysis as CA
        import comment_sentiment as CS

        def nums(line):
            return [int(m) for m in re.findall(r"(\d+)%", line)]

        # лексикон: три категории по одному комменту
        lex_rows = [{"id": "1", "parent": "root", "author": "a",
                     "text": "злая бешеная критика", "likes": 0},
                    {"id": "2", "parent": "root", "author": "b",
                     "text": "отличная работа, спасибо", "likes": 0},
                    {"id": "3", "parent": "root", "author": "c",
                     "text": "фу, не лучший", "likes": 0}]
        res = CA.analyze(lex_rows, use_neural=False, use_markers=False)
        line = CA.block(res).splitlines()[0]
        self.assertEqual(sum(nums(line)), 100,
                         f"блок печатает распределение, которое не складывается в 100: {line}")
        # прежний путь на тех же подсчётах обязан давать 99 — кейс различает методы
        counts = {}
        for r in lex_rows:
            cat, _ = CS.classify(r["text"])
            counts[cat] = counts.get(cat, 0) + 1
        old = sum(round(100 * v / len(lex_rows)) for v in counts.values())
        self.assertEqual(old, 99, "кейс обязан различать прежний путь round()")
        self.assertEqual(sum(CS.percentages(counts).values()), 100)

        # нейро-слой: три метки в трёх корзинах — та же обязанность
        def runner(texts, batch=32):
            labels = ["joy", "anger", "disappointment"]
            return {"emotion": [{l: 0.8} for l in labels],
                    "toxicity": [{"non-toxic": 0.9} for _ in texts],
                    "sarcasm": [{"neutral": 0.9} for _ in texts]}
        res2 = CA.analyze(lex_rows, runner=runner, use_markers=False)
        out2 = CA.block(res2)
        mood_line = out2.splitlines()[0]
        cross_line = [l for l in out2.splitlines() if l.startswith("Кросс-проверка")][0]
        self.assertEqual(sum(nums(mood_line)), 100,
                         f"нейро-распределение не складывается в 100: {mood_line}")
        self.assertEqual(sum(nums(cross_line)), 100,
                         f"кросс-проверка не складывается в 100: {cross_line}")

    def test_cross_check_only_when_the_layer_is_neural(self):
        """Кросс-проверка имеет смысл только когда доставляемые числа дал нейро-слой.
        Когда слой и есть лексикон, это второй проход по тому же словарю — мёртвая
        работа, и блок печатал «кросс-проверку», которая повторяет те же числа:
        читатель видит сверку, которой нет. Проверка поведением — spy-счётчик вызовов
        лексикона: на лексикон-пути обязан быть ОДИН вызов на комментарий (прежний код
        делал два — слой и сверку), на нейро-пути — один (сверка), и блок при лексиконе
        не печатает строку сверки."""
        import comment_analysis as CA
        import comment_sentiment as CS
        real = CS.classify
        calls = []

        def spy(text):
            calls.append(text)
            return real(text)

        rows = [{"id": "1", "parent": "root", "author": "a",
                 "text": "злая бешеная критика без фактов", "likes": 0}]
        CS.classify = spy
        try:
            res = CA.analyze(rows, use_neural=False, use_markers=False)
            self.assertIsNone(res["cross_check"], "лексикон сверяет сам себя")
            self.assertEqual(len(calls), 1,
                             "второй проход по тому же словарю")
            self.assertNotIn("Кросс-проверка", CA.block(res),
                             "блок выдаёт сверку, которой не было")
            calls.clear()
            res2 = CA.analyze(rows, runner=self.fake_runner(), use_markers=False)
            self.assertIsNotNone(res2["cross_check"],
                                 "сверка пропала там, где она нужна")
            self.assertEqual(len(calls), 1, "сверка обязана считать лексикон")
            self.assertIn("Кросс-проверка", CA.block(res2))
        finally:
            CS.classify = real

    def test_comment_path_goes_through_the_warm_worker_and_loads_once(self):
        """Правило: тёплый worker — один живой процесс на хозяина, и через него обязаны
        ходить ВСЕ пути. Раньше это выполнялось только для плагина: путь комментариев
        грузил три модели на каждый вызов — каждый анализ (каждый запуск скрипта —
        отдельный процесс) платил загрузку rubert заново. Проверка поведением: на
        worker-пути загрузка в процессе не вызывается вообще; когда worker не отвечает —
        один запасной путь, и модели загружаются ОДИН раз на все прогоны."""
        import comment_analysis as CA
        import worker_client as WC
        import neural_pass as NP
        real_client, real_infer = WC.client, NP.infer
        calls, loads = [], []

        class FakeClient:
            def call(self, op, **kw):
                calls.append(op)
                return {"ok": True, "scores": {n: [{n: 0.5}] for n in kw.get("models", [])}}

        class DeadClient:
            def call(self, op, **kw):
                return None

        def fake_loader(models):
            loads.append(tuple(models))
            return {m: (None, None, [m], False) for m in models}

        def fake_infer(tok, model, labels, texts, multi, name, batch):
            return [{"all": {labels[0]: 0.5}} for _ in texts]

        WC.client = lambda python=None: FakeClient()
        NP.infer = fake_infer
        try:
            runner = CA.neural_runner(loader=fake_loader)
            self.assertIsNotNone(runner, "путь комментариев не использует тёплый worker")
            runner(["текст1"])
            runner(["текст2"])
            self.assertEqual(calls, ["neural", "neural", "neural"],
                             "прогон прошёл не через worker")
            self.assertEqual(loads, [], "worker-путь грузит модели в своём процессе")
            # деградация: worker не поднялся — запасной путь, загрузка один раз
            calls.clear()
            loads.clear()
            runner2 = CA.neural_runner(client=DeadClient(), loader=fake_loader)
            self.assertIsNotNone(runner2, "нет запасного пути, когда worker не поднялся")
            runner2(["текст3"])
            runner2(["текст4"])
            self.assertEqual(loads, [tuple(CA.NEURAL_MODELS)],
                             "модели грузятся на каждый прогон, а не один раз")
        finally:
            WC.client = real_client
            NP.infer = real_infer



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
            # лексикон не сверяет сам себя: на лексикон-пути cross_check отсутствует
            # (прежний код делал второй проход по тому же словарю и печатал «сверку»,
            # которая повторяла те же числа)
            self.assertIsNone(lex["cross_check"],
                              "лексикон сверяет сам себя и выдаёт это как сверку")
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

    def test_comment_sentiment_text_output_has_no_nicknames(self):
        """Правило доставки (2026-10-18): блок комментариев — без ников и без цитат.
        Раньше текстовый вывод `comment_sentiment.py` печатал `@author «первые слова…»`
        по каждому треду — это тот вывод, который агент копирует в чат. Тест без сети:
        fake-строки подаются напрямую в форму вывода."""
        import comment_sentiment as CS
        rows = [{"id": "R1", "author": "Вася", "text": "злая бешеная критика без фактов",
                 "parent": "root", "likes": 0, "cat": "злость", "shouty": True},
                {"id": "R1.1", "author": "Петя", "text": "злая бешеная критика",
                 "parent": "R1", "likes": 0, "cat": "злость", "shouty": True}]
        res = CA.analyze(rows, use_neural=False, use_markers=False)
        out = CS.format_text(res, rows)
        for nick in ("Вася", "Петя"):
            self.assertNotIn(nick, out, f"ник {nick} попал в доставляемый вывод")
        self.assertNotIn("злая бешеная критика", out, "цитата комментария в доставляемом выводе")
        self.assertIn("злость 100%", out, "распределение по всем комментариям пропало")
        self.assertIn("однородность", out.lower(), "сводка по тредам пропала")
        self.assertIn("кричащих", out, "статистика по кричащим потерялась вместе с никами")

    def test_block_reports_unmeasurable_thread_as_not_measured(self):
        """Правило: тред с <2 содержательными — «не измеряется», а не None%;
        активные треды (≥2 содержательных) считаются и показываются числом."""
        rows = [{"id": "R1", "author": "a", "text": "злая бешеная критика", "parent": "root", "likes": 0},
                {"id": "R1.1", "author": "b", "text": "просто текст без позиции", "parent": "R1", "likes": 0}]
        res = CA.analyze(rows, use_neural=False, use_markers=False)
        out = CA.block(res)
        self.assertNotIn("None%", out, "блок печатает None% вместо «не измеряется»")
        self.assertIn("не измеряется", out, "тред с одним содержательным не назван неизмеримым")
        self.assertIn("Активных тредов (≥2 содержательных): 0 из 1", out,
                      "число активных тредов пропало из доставляемого блока")

        # измеримый тред (два содержательных) — число и процент на месте
        rows2 = [{"id": "R1", "author": "a", "text": "злая бешеная критика", "parent": "root", "likes": 0},
                 {"id": "R1.1", "author": "b", "text": "злая бешеная критика", "parent": "R1", "likes": 0}]
        res2 = CA.analyze(rows2, use_neural=False, use_markers=False)
        out2 = CA.block(res2)
        self.assertIn("Активных тредов (≥2 содержательных): 1 из 1", out2)
        self.assertIn("в среднем по тредам 100%", out2)

        # пустой набор — тоже без None%
        out3 = CA.block(CA.analyze([], use_neural=False, use_markers=False))
        self.assertNotIn("None%", out3, "пустой набор печатает None%")
        self.assertIn("не измеряется", out3)


# --- жизненный цикл worker: память и утечки (замер: 2.34 ГБ RSS на живой процесс) ---

def _alive(pid):
    return subprocess.run(["ps", "-p", str(pid)], capture_output=True).stdout.count(b"\n") > 1


def _start_worker(env):
    p = subprocess.Popen([sys.executable, os.path.join(SCRIPTS, "worker.py")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    hello = json.loads(p.stdout.readline().decode())
    return p, hello


class WorkerLifecycleTests(SnapshotMixin, unittest.TestCase):
    def test_worker_exits_on_idle(self):
        """Простой: worker не имеет права сидеть с моделями без запросов. Без выхода по
        простою живой процесс держит 2.34 ГБ RSS до конца своей жизни — на 16 ГБ машине
        это не тёплый кэш, а риск свопa."""
        env = dict(os.environ, PSYCH_WORKER_NO_MODELS="1", PSYCH_WORKER_IDLE="1")
        p, hello = _start_worker(env)
        self.assertEqual(hello.get("idle_s"), 1)
        # stdin НЕ закрываем: завершение должно быть именно по простою
        try:
            rc = p.wait(timeout=6)
        except subprocess.TimeoutExpired:
            p.kill()
            self.fail("worker не вышел по простою — он остаётся в памяти")
        self.assertEqual(rc, 0)

    def test_worker_exits_on_client_eof(self):
        env = dict(os.environ, PSYCH_WORKER_NO_MODELS="1", PSYCH_WORKER_IDLE="300")
        p, hello = _start_worker(env)
        p.stdin.close()
        try:
            rc = p.wait(timeout=6)
        except subprocess.TimeoutExpired:
            p.kill()
            self.fail("worker пережил EOF клиента — висячий процесс с моделями")
        self.assertEqual(rc, 0)

    def test_parent_death_detected(self):
        w = importlib.import_module("worker")
        real = os.getppid
        try:
            os.getppid = lambda: 1
            self.assertTrue(w.parent_changed(999))
        finally:
            os.getppid = real
        self.assertFalse(w.parent_changed(os.getppid()))

    def test_memory_guard_uses_current_rss_not_peak(self):
        """ru_maxrss — ПИК за жизнь процесса, он никогда не уменьшается: пиковый guard
        сработал бы один раз и остался «вечно превышенным» после освобождения моделей."""
        w = importlib.import_module("worker")
        real = w.current_rss_kb
        try:
            w.current_rss_kb = lambda: w.RSS_LIMIT_KB - 1000
            self.assertFalse(w.should_release(w.current_rss_kb(), w.RSS_LIMIT_KB))
            w.current_rss_kb = lambda: w.RSS_LIMIT_KB + 1000
            self.assertTrue(w.should_release(w.current_rss_kb(), w.RSS_LIMIT_KB))
        finally:
            w.current_rss_kb = real

    def test_memory_guard_releases_all_layers_and_reports(self):
        w = importlib.import_module("worker")
        state = {"embedding": object(), "classifier": object(),
                 "neural": {"emotion": object()}, "matrix": None}
        real = w.current_rss_kb
        seq = [w.RSS_LIMIT_KB + 1000, w.RSS_LIMIT_KB - 500]   # RSS до / после освобождения
        try:
            w.current_rss_kb = lambda: seq.pop(0)
            freed = w.release_models(state)
            # Освобождается ВСЁ, а не только HF-модели: fastembed с ONNX сами держат
            # ~1.8 ГБ, и прежний guard отдавал только neural, оставляя эти гигабайты —
            # читатель видел «освобождение», которого не было
            self.assertEqual(state, {"embedding": None, "classifier": None,
                                     "neural": None, "matrix": None})
            self.assertEqual(freed, 1500)
            self.assertEqual(w.release_models(state), 0)   # нечего освобождать — не делит на ноль
        finally:
            w.current_rss_kb = real

    def test_guard_silent_when_only_the_matrix_is_loaded(self):
        """Косинус-матрица — не модельный слой. Прежде `any(state.values())` считала её
        «чем-то загруженным»: guard срабатывал на пике RSS, писал `released_kb: 0` —
        освобождение, которого не было — и ставил флаг навсегда, то есть один пик убивал
        нейро-слой до конца жизни долгоживущего процесса гейтвея."""
        w = importlib.import_module("worker")
        state = {"embedding": None, "classifier": None, "neural": None,
                 "matrix": ([1], [2])}
        resp = {"ok": True, "results": []}
        w.current_rss_kb = lambda: w.RSS_LIMIT_KB + 1000
        self.assertFalse(w.guard_memory(resp, state))
        self.assertNotIn("memory", resp)
        self.assertIsNotNone(state["matrix"])       # guard не трогает то, что не модель

    def test_guard_says_when_release_returned_nothing(self):
        """Освобождение произошло, но RSS не упал — аллокатор macOS страницы ОС не отдаёт.
        Заметка обязана сказать это прямо, а не подать 0 КБ как эффективное освобождение."""
        w = importlib.import_module("worker")
        state = {"embedding": object(), "classifier": None,
                 "neural": {"x": object()}, "matrix": None}
        resp = {"ok": True, "markers": []}
        w.current_rss_kb = lambda: w.RSS_LIMIT_KB + 1000
        self.assertTrue(w.guard_memory(resp, state))
        self.assertEqual(resp["memory"]["released_kb"], 0)
        self.assertTrue(resp["memory"]["rss_did_not_drop"])
        self.assertTrue(all(v is None for v in state.values()))

    def test_guard_counts_rss_once_per_request(self):
        """Проверка RSS — ОДИН раз на запрос, и это же значение передаётся в release_models
        как «до». Прежний код снимал RSS дважды для одной проверки (3.3 мс на вызов —
        лишние ~7 мс на сообщение в горячем пути, где тёплый путь мерился как 84 мс);
        всего на сработавший запрос остаётся проверка плюс замер «после» gc."""
        w = importlib.import_module("worker")
        state = {"embedding": object(), "classifier": None, "neural": None, "matrix": None}
        resp = {"ok": True, "markers": []}
        calls, seq = [], [w.RSS_LIMIT_KB + 1000, w.RSS_LIMIT_KB - 500]

        def fake():
            calls.append(1)
            return seq.pop(0) if seq else w.RSS_LIMIT_KB
        w.current_rss_kb = fake
        self.assertTrue(w.guard_memory(resp, state))
        self.assertEqual(len(calls), 2)          # проверка + замер после gc, не дважды для проверки
        self.assertEqual(resp["memory"]["rss_kb"], w.RSS_LIMIT_KB + 1000)
        self.assertEqual(resp["memory"]["released_kb"], 1500)
        self.assertFalse(resp["memory"]["rss_did_not_drop"])

    def test_neural_gate_released_beats_empty_state(self):
        """Пустой dict после освобождения читается как «загрузить заново» — это вернуло бы
        те же гигабайты, ради которых их освободили. Флаг released обязан быть важнее."""
        w = importlib.import_module("worker")
        self.assertEqual(w.neural_gate({}, True), "released")
        self.assertEqual(w.neural_gate({}, False), "load")
        self.assertEqual(w.neural_gate({"emotion": object()}, False), "run")

    def test_released_guard_does_not_reload_models_in_caller(self):
        ca = importlib.import_module("comment_analysis")
        loads = []

        class FakeClient:
            def call(self, op, **kw):
                return {"ok": False, "released": True}

        runner = ca.neural_runner(client=FakeClient(),
                                  loader=lambda models=None: loads.append(1) or None)
        r = ca.analyze([{"id": "1", "parent": "root", "author": "a", "text": "отлично"}],
                       runner=runner)
        self.assertEqual(loads, [], "caller перезагрузил модели, хотя worker отдал память")
        self.assertTrue(r.get("neural_released"))
        self.assertEqual(r["layer"], "lexicon")
        self.assertIn("memory guard", ca.block(r))

    def test_memory_guard_appends_to_the_same_response(self):
        """Один ответ на один запрос. Отдельный emit сдвигает ответы клиента: следующий
        запрос читает строку предыдущего — живая проверка показала, что ответ на `markers`
        пришёл как `neural released`. Guard обязан добавлять память к тому же dict."""
        w = importlib.import_module("worker")
        real = w.current_rss_kb
        try:
            w.current_rss_kb = lambda: w.RSS_LIMIT_KB + 1000
            resp = {"ok": True, "scores": {}}
            state = {"embedding": None, "classifier": None,
                     "neural": {"emotion": object()}, "matrix": None}
            self.assertTrue(w.guard_memory(resp, state))
            self.assertIn("memory", resp)
            self.assertTrue(all(v is None for v in state.values()))
            resp2 = {"ok": True, "markers": []}
            self.assertFalse(w.guard_memory(resp2, state))   # нечего освобождать — guard молчит
            self.assertNotIn("memory", resp2)
        finally:
            w.current_rss_kb = real

    def test_default_limit_is_above_steady_state_rss(self):
        """Штатный RSS загруженного worker — 3.39 ГБ (замер на этой машине: fastembed +
        ONNX + три HF-модели в одном процессе; 2.34 ГБ — только HF-модели). Если лимит
        ниже, guard срабатывает на первом запросе и нейро-слой никогда не работает:
        живая проверка показала, что числа получил словарь."""
        w = importlib.import_module("worker")
        self.assertGreater(w.RSS_LIMIT_KB, 3_390_000)

    def test_loop_emits_once_per_request(self):
        """Guard-проверка (не поведенческая): ops обязаны возвращать dict, а не emit —
        иначе guard пишет вторую строку и ответы клиента разъезжаются."""
        src = read(os.path.join(SCRIPTS, "worker.py"))
        loop = src[src.index("while True:"):]
        self.assertIn("resp = dispatch(req)", loop)
        self.assertEqual(loop.count("emit("), 4,
                        "в цикле должны emit-иться только guard-строки (idle, parent, bad json) и resp")

    def test_idle_release_and_pending_note(self):
        """Память на паузе: долгий хозяин (гейтвей) живёт часами, и 3.4 ГБ модельного RSS
        в нём — занятая память. Модели освобождаются, процесс остаётся, следующий запрос
        вправе загрузить заново (это не guard). Заметка едет в ответе на запрос, а не
        отдельной строкой — отдельная строка сдвинула бы ответы клиента."""
        w = importlib.import_module("worker")
        self.assertTrue(w.idle_release_due(0, 130, 120))
        self.assertFalse(w.idle_release_due(0, 119, 120))
        resp = {"ok": True, "markers": []}
        self.assertFalse(w.attach_pending(resp, None))
        self.assertTrue(w.attach_pending(resp, {"idle_release": True, "released_kb": 900_000}))
        self.assertIn("memory", resp)
        # после освобождения по простою перезагрузка разрешена: released (guard) не ставится
        self.assertEqual(w.neural_gate({}, False), "load")

    def test_client_registers_atexit(self):
        """Выход процесса-хозяина обязан забирать worker с собой."""
        wc = importlib.import_module("worker_client")
        calls = []
        real = wc.atexit.register
        try:
            wc.atexit.register = lambda fn, *a: calls.append(fn)
            c = wc.WorkerClient(python=sys.executable)
            self.assertEqual(calls, [c.stop])
        finally:
            wc.atexit.register = real


if __name__ == "__main__":
    unittest.main(verbosity=2)
