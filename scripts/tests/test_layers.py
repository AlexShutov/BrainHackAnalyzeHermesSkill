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
import classifier

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
        """Правило: правило END_MAP должно быть достижимо классом слов."""
        for src, _ in side.END_MAP:
            self.assertGreaterEqual(len(src), 3, f"правило «{src}» недостижимо при классе {{3,}}")

    def test_side_share_has_a_threshold(self):
        self.assertTrue(hasattr(side, "SIDE_THRESHOLD"),
                        "docstring обещает «≥ порога», порога в коде нет")

    def test_side_share_and_ukr_detect_agree_on_short_fragment(self):
        frag = "маніпуляцыя"
        thr = getattr(side, "SIDE_THRESHOLD", None)
        self.assertIsNotNone(thr, "порога в коде нет — сравнивать не с чем")
        self.assertEqual(ukr.lang_of(frag), ("ru" if side.side_share(frag) < thr else "uk"),
                         "два слоя дают разные вердикты на одном фрагменте")

    def test_status_post_guard_length_and_flood(self):
        src = read(os.path.join(YT, "status_post.py"))
        self.assertIn("4096", src, "нет контроля 4096 символов")
        self.assertIn("HTTPError", src, "нет обработки 429/сетевой ошибки")

    def test_tie_order_is_one_for_both_paths(self):
        """Один лексикон — один порядок ничьи."""
        src = read(os.path.join(YT, "telegram_comments.py"))
        self.assertIn("TIE_ORDER", src, "telegram-путь решает ничью по порядку словаря")

    def test_neural_labels_have_no_none_and_known_models(self):
        src = read(os.path.join(YT, "neural_pass.py"))
        self.assertIn("REPOS[name]", src)
        self.assertTrue(set(neural_pass.MULTILABEL) >= set(neural_pass.REPOS))
        self.assertIn("None not in labels", src, "нет защиты от None-метки в id2label")

if __name__ == "__main__":
    unittest.main(verbosity=2)
