---
name: bigbro-analyze
description: "Use when analyzing text/messages with the video tag system."
version: 1.0.0
author: Алексей (via Hermes)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Analysis, Text, Telegram, Manipulation, Marking]
    related_skills: [youtube-content, voice-transcription]
---

# Text / message analysis (same system as video analysis)

## When to use

The user sends, forwards or pastes text — a Telegram message, a forwarded channel post, a screenshot of a post, a chat fragment, an article, a document, a voice message — and asks for analysis (or the analysis is implied by the standing preference). Apply the same tag system as `youtube-content`; load that skill's marking table if the tags are not in context. AUTO-TRIGGER: any multi-paragraph text (more than one paragraph) sent for analysis runs this skill without being asked. Single-paragraph messages and one-line questions do not.

## Output format

1. **Source line** first: the link (if any), otherwise `источник: <chat/channel/author>, <date>` — for forwarded posts name the original author/channel, not the forwarder.
2. **Summary by thought** — split the text into logically complete thoughts (one idea per unit: a claim + its support, a remark, a conclusion). For EACH thought: its indicator emoji, then a 1–3 sentence summary of what that thought says. No timestamps and no paragraph numbers for text — the thought itself is the locator (quote 2–5 words of it if needed to identify).
3. Order = order of appearance in the text, one block per thought, never grouped by tag. If a thought mixes categories, put the primary indicator on its own line and note the others inside the summary. If the source has real timestamps (video/screenshot with timecodes), keep the `EMOJI [MM:SS–MM:SS]` format — timestamps are for video only.
4. Deliver in chat/Telegram. Never write files unless the user explicitly asks. Telegram caps one message at 4096 chars: measure the assembled answer programmatically, shorten the longest thought descriptions first, and only split into parts if still over — then the chat reply is a short confirmation, not a repeat of the parts (same rule as `youtube-content`).

## Marking system (user-defined, standing preference)

| Tag | Category |
|-----|----------|
| 🔴 | Критика России: армии, правительства, институтов власти; скрытый подтекст, критикующий Россию; манипулятивные сопоставления, выставляющие Россию в плохом свете. Если о России ничего негативного нет — 🔴 для общего негатива/токсичности |
| 🍆 | Явная анти-российская пропаганда — сильнее простой критики; обычная критика остаётся 🔴 |
| 🍏 | Позитив о России: прогресс, успехи, победы, подтверждённые фактами |
| 🐻 | Необоснованное про-российское восхваление: «гойда», гордость «мы великие», хвалебные оценки ВС РФ и России без конкретных фактов, приёмы про-российской манипуляции (апелляция к гордости/страху, подвиг без доказательств, «наши» vs «не наши», замалчивание плохого в России при подчёркнутом плохом на Западе/Украине) |
| 🟢 | Прогресс человечества: наука, производство, роботы, ИИ, мировая инфраструктура |
| 🟤 | Реклама: отдельные блоки и нативная (спонсорские упоминания, промо продуктов/медиа, продвижение своих продуктов). Простой самопиар («подписывайтесь») — не реклама. Если рекламы нет — сказать явно |
| 🦹 | Инфоцыганщина в экономическом тексте: тезисы, противоречащие экономической теории (гарантированная доходность, «рынок всегда растёт», «дивиденды = гарантированная прибыль», высокая доходность без риска, ошибка выжившего), и продажа своих курсов/рассылок/консультаций по экономике и фондовому рынку |
| 🟣 | Фрагменты вне общего контекста (отступления, шутки, мета-разговор), не реклама и без анти-российского подтекста |
| ⚫ | Психологические манипуляции: подмена понятий, софистика, логически неверные импликации, ложные дилеммы, апелляция к эмоциям/авторитету, переворот причинно-следственных связей, ошибка выжившего, post hoc. Явно называть приём и объяснять, в чём именно ошибка |
| ⭐ | Обучающая техническая информация и технические инсайды: объяснение, как что-то устроено/работает, и инсайдерские детали — то, что рассказал источник изнутри, чего нет в публичных данных |
| 🐷 | Явное расхваливание Украины, ВСУ и ударов по России: похвала ударов/атак, оправдание обстрелов, «молодцы», «правильно делают», «так и надо», позитивная оценка разрушений. Только явная похвала (в том числе процитированная) |
| 🦅 | Необоснованное манипулятивное восхваление США и западного мира: Запад как эталон без аргументов, двойные стандарты, идеализация институтов/общества, апелляция к «цивилизованности» |
| 🔵 | Скрываемая тайна / «правда», которую раскрывают: «на самом деле», «настоящая причина», инсайдерские знания, замалчиваемые факты |

## RAG (psychology index)

Before marking, query the local psychology index — one call per analysis:

- MCP tool `psychology_search(query, top_k)` (server `psychology-rag`, stdio, venv `~/.hermes/tools/psych-rag-env`), or the CLI `scripts/rag_query.py "<text>"`.
- Corpus: `psychology/devices.json` (the closed device list — draft, 70 entries: 36 logical + 14 Dilts language focuses + 11 внушение + 9 сквозные/уровень текста) + `psychology/psychology/…fb2` book chunks (277 chunks, 1200 chars / 200 overlap, section titles kept as labels). Storage: SQLite FTS5 + vectors, brute-force cosine, hybrid `0.4·BM25 + 0.6·cosine`, +0.15 boost for exact device-name hits, top_k 6, min_score 0.30, output ≤2500 chars.
- Measured (M1 Pro, fastembed ONNX `paraphrase-multilingual-MiniLM-L12-v2`, 384 dims): warm model load 0.5 s, index build 6.9 s, query 5–20 ms. The MCP server keeps the model loaded, so per-call cost is milliseconds; the CLI path pays ~0.6 s model load per call.
- BM25-only fallback (`--no-embed`, pure stdlib, ~0.3 ms) when ONNX cannot run — results are lexical-only, the skill still works.
- The `psych-rag` plugin (`pre_gateway_dispatch`) appends three blocks to multi-paragraph inbound messages automatically: the retrieved passages, the marker hits from `scripts/markers.py`, and — only when markers fired — the classifier's second opinion. Single-paragraph messages are not rewritten (matches the auto-trigger rule above).
- Retrieval supplements the list, it never replaces it: a device that exists in the list but not in the corpus must still be applied from the list.

## Rules

- Источники для ⚫: закрытый список приёмов — `references/manipulation-devices.md`; корпус (книга по манипуляциям в `psychology/` — FB2 + извлечённый текст) опрашивается retrieval-ом по правилам `local-rag-on-weak-hardware`. Статическая карта списка подаётся всегда, retrieval добавляет определения и примеры из корпуса — так список не «плывёт» от запроса к запросу.
- ⚫ requires an explicit call-out: name the device and the specific error («подмена: X выдаётся за Y», «ложная импликация: из A не следует B», «ложная дилемма», «апелляция к страху», «ошибка выжившего»). Если фрагмент одновременно содержательный и манипулятивный — ⚫ в таймлайне, другая категория в описании.
- Двухслойная детекция: текст не обязан содержать название приёма или слова из корпуса. Слой 1 — `scripts/markers.py` + `psychology/markers.json` (детерминированный, без LLM): триггеры пресуппозиции по Levinson (definite descriptions, factive/implicative/change-of-state verbs, iteratives, verbs of judging, temporal clauses, clefts, comparatives, counterfactual conditionals, wh-questions), модальность и эпистемическая дистанция (absolutes, hedging, evidentiality gap), номинализация и удаление агента, оценочная/эмоциональная лексика, структурные маркеры (imperative+benefit, benefit to speaker, disclaimer, narrative markers, universal from single case, repetition, question trap). Слой 2 — retrieval даёт определение приёма. Вывод строится из маркера + определения, а не из совпадения слов с RAG: маркер показывает, что именно в тексте делает фразу приёмом.
- Слой 3 — независимое второе мнение: `scripts/classifier.py` (XLM-RoBERTa-large, 23 метки SemEval-2023 Task 3, ONNX int8 `psychology/models/classifier_int8.onnx`, пороги `thresholds.npy`, перевод меток через `psychology/taxonomy_map.json`). Русский вход идёт напрямую, переводить текст на другой язык нельзя — перевод разрушает лексические доказательства слоя 1. Гейчится слоем 1: запускается только на абзацах, где маркеры сработали. Стоимость: загрузка 0.5 с, 19 мс на абзац в тёплом процессе, пик ~1.3 ГБ RAM. Классификатор не авторитет (fine-grained F1 ~0.45): он никогда не перекрывает закрытый список и маркеры; при расхождении показывать оба («классификатор видит X, по маркерам это Y»), цитата-доказательство из маркерного слоя обязательна. Подробности и замеренное сравнение кандидатов — `references/classifier-layer.md`.
- Версия модели закреплена в `psychology/models/model_version.json` (id + revision + формат). При первом использовании в сессии `scripts/model_update_check.py` сравнивает записанную версию с hub и ищет схожие модели с лучшим заявленным метриком, покрывающие русский. Если нужно обновление — состояние помечается как нужное (`update_state.json`), загрузка идёт в фоне (`model_update_run.py`), новая модель проверяется на трёх контрольных текстах. Текущая модель продолжает работать, пока новая не проверена; молча менять модель нельзя — продвижение только явное (`model_promote.py`). Если обновление начато, сказать это явно: «начато обновление психологической модели, продолжается использование старой».
- Сквозной проход обязателен после по-мыслового: отдельный блок «Сквозной разбор» для приёмов, которые существуют только в сочетании абзацев (подводка/priming, накопительный фрейминг, отложенный вывод, нарративная дуга, эмоциональная нарастающая, преёмпт-инокуляция, апелляция к здравому смыслу, селекция на уровне текста, повтор с вариацией). Приём, существующий только в сочетании абзацев, не приписывается одному абзацу; абзац-подводка маркируется как подводка с указанием, какой вывод она готовит. Для каждого сквозного приёма перечислить участвующие мысли (2–5 слов цитаты из каждой) и вывод, который они вместе строят. Gate-тест применяется на уровне текста: если убрать рамку, вывод перестает следовать.
- Gate for ⚫: strip the emotional/rhetorical frame and leave only the facts — if the conclusion no longer follows, it was induced by a device, so ⚫; rhetoric without a conclusion is not ⚫. Device names and the working closed list: `references/manipulation-devices.md` (confirm scope with the user before treating it as exhaustive).
- Разграничения: 🍏 (факты) vs 🐻 (без фактов, эмоционально); 🦅 (идеализация без аргументов) vs нейтральное сравнение с цифрами; 🐷 (явная похвала) vs нейтральное сообщение об ударах; 🦹 (тезис против теории + продажа курса) vs корректный экономический анализ; 🟤 (реклама) vs самопиар; нейтральная политика (законы, выборы как факты) не маркируется.
- Ложное или одностороннее утверждение само по себе не ⚫ — ⚫ только за сам логический приём.
- Факты в тексте не выдумывать: если автор что-то утверждает без источника, так и писать («без подтверждения»); проверяемые утверждения проверять и указывать, подтверждены ли они. Сокращения и инициалы автора (ВБ, ЗГТ, аббревиатуры компаний) читать как написано; если раскрытие неоднозначно — отдельной строкой указать, что это допущение, а не факт.
- Закрывать разбор строкой «Нет <метки>» — перечислить категории, которых в тексте не оказалось; это отличает «не было» от «не заметил».
- Метки — общий словарь с `youtube-content`: при изменении определения метки обновлять оба скилла, иначе текст и видео начнут размечаться по-разному.
- Скриншоты: читать через vision_analyze, цитировать формулировки автора дословно там, где это важно для оценки.
- Голосовые: расшифровка через voice-transcription, затем тот же разбор.
- Если в тексте нет ни одного маркируемого фрагмента — сказать об этом явно, а не придумывать метки.
- Записи о запросах идут в Obsidian vault (`База знаний о сообщениях/Сообщения/YYYY-MM-DD.md`) только если пользователь просит; сам разбор — в чат.
