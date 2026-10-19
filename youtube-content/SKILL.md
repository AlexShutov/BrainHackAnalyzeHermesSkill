---
name: youtube-content
description: "Transcripts to tagged analyses: manipulation, ads, comments. Neural comment layer through one entry point on all three paths."
version: 2.5.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [YouTube, Video, Transcripts, Media]
    related_skills: []
---

# YouTube Content Tool

## When to use

Use when the user shares a YouTube URL or video link, asks to summarize a video, requests a transcript, or wants to extract and reformat content from any YouTube video. Transforms transcripts into structured content (chapters, summaries, threads, blog posts).

## Setup

Use `terminal` with the Python from a PM-prepared Hermes source checkout. The
`youtube` extra declares the helper's dependency; do not install packages into
Hermes with raw pip or project-discovering `uv run`.

From that checkout, first follow the isolated development-home setup in
[Package Management](https://hermes-agent.nousresearch.com/docs/reference/package-management#developer-workflow),
then prepare the extra and reactivate before running the helper:

```bash
source ./activate
python -c "import pm; pm.sync_venv(['youtube'], explicit=True)"
source ./activate
python -c "import youtube_transcript_api; print(youtube_transcript_api.__file__)"
```

On Windows, use `. .\activate.ps1` instead of `source ./activate`. If the terminal
runs on a different host or in a sandbox, use an explicitly isolated helper
environment there, not the agent's production environment. Run every command
below with the interpreter whose import check succeeded.

### Fallback environment (no PM checkout available)

If there is no Hermes source checkout to sync the `youtube` extra from, install
the dependency into an ISOLATED target dir and point PYTHONPATH at it — never
raw pip into the Hermes environment:

```bash
python -m pip -q install --target ~/.hermes/cache/scratch/yt-pylib youtube-transcript-api
PYTHONPATH=~/.hermes/cache/scratch/yt-pylib python SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only --timestamps
```

Pitfall: the helper uses PEP 604 annotations (`list | None`), so it needs
Python >= 3.10. Stock macOS `python3` is 3.9 and dies with
`TypeError: unsupported operand type(s) for |` — run it with the Hermes
interpreter (`python`) or a Homebrew python3.

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md file. The script accepts any standard YouTube URL format, short links (youtu.be), shorts, embeds, live links, or a raw 11-character video ID.

```bash
# JSON output with metadata
python SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Plain text (good for piping into further processing)
python SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# With timestamps
python SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# Specific language with fallback chain
python SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## Rutube (same analysis, different plumbing)

For `rutube.ru/video/<32-hex-id>/` links the marking system, sorting rules, delivery format and the
log record are identical — only the plumbing differs:

- **Transcript**: no transcript API. Use `scripts/rutube_transcript.py URL` (yt-dlp `rutube`
  extractor, `--write-sub --sub-format srt`). Auto-generated `ru` tracks exist on many videos but
  NOT on all — when the extractor reports "has no subtitles", say so plainly and never fabricate.
  Fallback when there are no captions: transcribe with faster-whisper — it is a machine transcript,
  so flag that in the answer. Long videos are still worth transcribing, not skipped (~10× realtime,
  ~1 GB RAM); plumbing details in `references/rutube.md`.
- **Comments**: yt-dlp returns `comments: None` for Rutube. Use `scripts/rutube_comments.py URL` —
  it reads the public API `https://rutube.ru/api/comments/video/<id>/?limit=100` (paginate via
  `next`) and reuses the same lexicon as `comment_sentiment.py`, so mood + thread-homogeneity
  numbers are comparable across platforms. The API's `comments_count` can exceed the number of
  fetchable comments (deleted/hidden) — report the fetched count, not the declared one.
- **Titles**: the Rutube API `https://rutube.ru/api/video/<id>/` returns `title`/`author`; oEmbed
  does not work for Rutube.
- Auto-subtitles garble names and brands (same caveat as YouTube auto-subs).

## Telegram post comments (user session, lazy login)

Trigger: the user forwards a post and asks about its comments (explicitly, or right after pressing «Показать»). The Bot API cannot read discussion threads — comments are fetched through the user's OWN Telethon session, stored LOCALLY on the machine where this bot runs; each colleague's session lives in their own instance, sessions are never shared.

- Extract the origin from the forward: chat (username or `-100…` id) + post id; a `t.me/<channel>/<id>` or `t.me/c/<id>/<id>` link works directly via `--link`.
- Parsing rules (mechanics, verified): query and fragment are cut before parsing (`?t=123` no longer lands in the post id); only the path segment `s` is stripped (preview form `t.me/s/<channel>/<id>`) — never the end of a channel name. For `t.me/c/<internal>/<id>` the internal channel id is returned as an int, and `get_entity` needs `PeerChannel(внутренний id)`: `«-100<id>»` is the Bot API convention, Telethon reads negative ids by its own scheme (`marked = -(1e12 + id)`) and would understand `-1001543` as chat 1001543, not channel 1543. A `--chat` string `«-1001543»` is converted via `peer_for()`. Bad input returns `bad_input`, not a crash.
- Run: `psych-rag-env/bin/python scripts/telegram_comments.py comments --link <url> --sentiment` (telethon is installed in the psych-rag venv). Returns all comments (author, text, date, replies, reactions) plus the sentiment block — mood + per-thread homogeneity computed by the SAME lexicon as `comment_sentiment.py`; deliver it in the same shape as the YouTube comments block.
- **`--limit 0` in Telethon means ZERO messages, not "all"** — `iter_messages` ends immediately when `limit <= 0` (`RequestIter`: `self.limit = max(float('inf') if limit is None else limit, 0)`), so the default path returned an empty list and an empty thread looked like "no comments". Fixed 2026-10-19: the script maps `0 → None` («все» — это None). Before telling the user a post has no comments, check the limit semantics — an empty result from a `limit=0` call is not evidence of an empty thread. Tie-breaking in this path follows the lexicon dict order, not `TIE_ORDER` — the two paths can disagree on a tie; state which path produced the numbers.
- **If it returns `login_required` — lazy login dialogue, run BEFORE anything else and ALWAYS with an explicit explanation.** Say why: Telegram Bot API cannot read comments, so we read the discussion thread through the user's own account; the session file stays on this machine only, the login code arrives on the user's phone, and each bot instance (each colleague) logs into their own account on their own machine. NEVER log in during skill installation — only here, on first need. Then walk the steps:
  1. one-time `api_id`/`api_hash` from my.telegram.org (free);
  2. phone number;
  3. the code Telegram sends (and cloud password if set) — run `login` again with `--code` (`--password`).
- Honest caveats: auto-generated names are as-is; deleted/hidden comments reduce the fetched count vs the declared one — report the fetched count; lexicon sentiment misclassifies sarcasm.

## Comments (no API key needed)

Comments are fetchable WITHOUT a YouTube Data API key via the innertube endpoint through yt-dlp. The official Data API (`commentThreads`) requires an API key — not configured here; do not guess one. Install yt-dlp into the same isolated target dir as the transcript lib:

```bash
python -m pip -q install --target ~/.hermes/cache/scratch/yt-pylib yt-dlp
PYTHONPATH=~/.hermes/cache/scratch/yt-pylib python SKILL_DIR/scripts/fetch_comments.py "URL" [--limit N] [--json]
```

Fields per comment: `author, text, likes, timestamp (unix), pinned, parent` (`parent == "root"` = top-level; otherwise the parent comment id — replies). Order is YouTube's default (top). Pitfalls:
- Comments disabled on the video → script exits with a clear message; say so, don't fabricate.
- Fetching ALL comments on a big video is slow — use `--limit` for a quick pass (e.g. top 50), and cache the fetched set by video id in scratch so a second analysis of the same video reuses it instead of re-fetching.
- The WEB innertube client now returns key-based commentViewModels without text (raw scraping returns 0 comments); yt-dlp handles this correctly — do not hand-roll innertube scraping.
- Comments are part of the standard analysis: on any «анализ»/«проанализируй» request, fetch ALL comments and run `scripts/comment_sentiment.py` — deliver the mood + homogeneity block after the timeline (see below). For plain summary/transcript requests without the trigger word, comments are not fetched.

## Language gate (ukr_detect.py) — mechanics after the 2026-10-18 fix

- Evidence is counted ONCE PER WORD (not per character, and not twice as letter+dictionary). A short entry (≤3 chars: `ще`, `дід`, `є`, `па`, `ды`) matches only as a whole word — otherwise the Russian `щелей` reads as Ukrainian `ще`. Longer entries match as a prefix so inflected forms are caught (`українські`, `найкращий`, `паняццях`).
- Weak entries (homonyms that also exist in Russian: `па`, `ды`, `якая`, `які`, `таму`, `аб`, `малы`) never decide a language on their own — they only reinforce a language already decided by strong evidence. `AMBIG` entries are never counted as unambiguous (they must not also sit in `UK_WORDS`).
- One DECISIVE evidence decides the language: `ї, є, ґ` (absent in Belarusian and Russian) → uk; `ў` (absent in Ukrainian) → be. Shared `і`/`як`/`але` only prove "not Russian" → `amb`.
- Majority gate is measured by TEXT VOLUME (`nonru_share_chars`), not by the number of comments — `nonru_share_items` is reported separately for cross-check. One short Ukrainian comment among long Russian ones is 2% of text, not 50%.
- 🐷 is set only when the segment language is `uk` (`tag` field); `be` and `amb` get `tag=None` and are processed as usual Russian.

## Comment mood + thread homogeneity (standing algorithm)

**Neural models (standing, replaces the dictionary lexicon).** Run on every analysis with the psych-rag-env interpreter (`/Users/alex/.hermes/tools/psych-rag-env/bin/python`, transformers + torch available). Always pass `truncation=True, max_length=512` — comments longer than 512 tokens crash BERT position embeddings (`RuntimeError: size of tensor a (625) must match tensor b (512)`). Ярлыки берутся из `config.id2label` самой модели (`neural_pass.py`); жёсткий список — только fallback для оффлайн-режима без config.json, и он обязан совпадать с порядком id2label: несоответствие порядка молча выдаёт другую эмоцию. Multi-label модели — sigmoid, одиночный класс — softmax.

- `seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions` — 28 multi-label emotions (anger, annoyance, disapproval, disgust, disappointment, sadness, grief, nervousness / admiration, joy, amusement, approval, love, gratitude, optimism, excitement, pride, relief, caring, desire / neutral). Test F1 ~0.79 macro on CEDR, AUC 0.93.
- `cointegrated/rubert-tiny-toxicity` — multi-label: non-toxic, insult, obscenity, threat, dangerous (per-label ROC AUC: insult 0.991, threat 0.991, obscenity 0.988).
- `Kostya165/rubert_tiny2_russian_emotion_sentiment` — aggression / anxiety / neutral / positive / sarcasm (sarcasm recall ~0.95). Use it for the sarcasm pass — the dictionary lexicon cannot see irony, which is why it over-reports «нейтрально».
- Reasoning devices in comments: `scripts/classifier.py` with `min_conf` per-label thresholds from `thresholds.npy` (calibrated per label — this is the correct statistical criterion; a fixed p is only a rough proxy). Require a named device, not just loaded vocabulary: `Loaded_Language` alone is NOT a manipulation.

Emoji mapping for comments (one tag per comment): 💸 — продающий/маркетинговый комментарий (CTA, дефицит, «скидка только сегодня», реклама товара в тексте комментария); 🎭🧠 / 📢🧠 — уверенный вердикт по приёму рассуждения; ☠️ — смерть/суицид (смерть техники не считается); 🤮 — сарказм (по выбору пользователя, уточняется); 🔴 — оценочный негатив без фактической базы (токсичность/оскорбление/агрессия по модели или негативная эмоция); 😊 — позитив; 🙂 — нейтрально. Приоритет: 💸 → приём → ☠️ → 🤮 → 🔴 → 😊 → 🙂.

**Delivery format for the comment block:** no quoted comment text, no usernames (правило пользователя, 2026-10-18 — ники случайных комментаторов не показывать). Два блока: (1) распределение по ВСЕМ комментариям (counts + %) плюс одна строка-оговорка про модель; (2) сводка по тредам БЕЗ перечисления ников: число активных тредов (≥2 содержательных), распределение доминирующих тегов по тредам, средняя однородность, и 1–2 предложения о самых дружных темах по содержанию. Threads with only neutral comments are not counted as active. State once that the classifier is a model, not a dictionary: sarcasm/irony can still be misclassified, and homogeneity measures emotional coloring, not semantic agreement.

Legacy lexicon path (`scripts/comment_sentiment.py`) remains for cross-checks. **One entry point for the comment block, one layer on all three paths** (YouTube, Rutube, Telegram): `scripts/comment_analysis.py` — the single input module for the comment block. It takes the canonical comment form (`id, parent, author, text, likes, timestamp, pinned`), runs the neural pass, maps the model's 28 emotion labels onto the delivered baskets (радость/злость/недовольство/нейтрально), keeps the lexicon as a cross-check, and applies the tag priority (💸 → приём → ☠️ → 🤮/🤢 → 🔴 → 😊 → 🙂). `comment_sentiment.py`, `rutube_comments.py` and `telegram_comments.py` all call it — the Telegram path used to run lexicon-only, which made numbers incomparable across platforms. Threads come from `build_threads` (single definition: root + all descendants, reply-on-reply stays in the same thread). `--no-neural` forces the lexicon for comparison; the delivered block always names which layer produced the numbers. If a path can only run the lexicon, name which layer produced the numbers instead of mixing them.

Run the neural pass in batches (batch 32), not comment by comment — on a few hundred comments per-item inference is ~10× slower.

Lexicon matching semantics (fixed 2026-10-18, was substring-only): stems match at word start, short stems (≤3 chars: `ура`, `бык`, `фу`) only as whole words — otherwise `ураган` reads as радость and `классический` as радость (`класс` was removed from the lexicon for this reason). Negation flips a positive-category hit: `не лучший`, `не радость`, `не злой` → недовольство, not радость/злость. Negated forms whose base stem lives in JOY/ANGER are NOT listed in DISCONTENT (double counting). Tie between categories resolves to the negative reading (order: злость → недовольство → радость), deterministically.

## Ukrainian / Belarusian language pass (правило пользователя, 2026-10-18)

Trigger: any analysis — comments and transcript are checked for «мова» (both Ukrainian and Belarusian are called мова).

- **Gate (deterministic, no LLM):** `scripts/ukr_detect.py` — two layers:
  - letters that decide the uk/be dispute: **ї, є, ґ** (Belarusian has none of them) vs **ў** (Ukrainian has none). The shared **і** does NOT decide the language by itself.
  - words: `UK_WORDS` = forms that differ from Russian AND from Belarusian; `BE_WORDS` = Belarusian forms that differ from Russian (гэта, шмат, вельмі, каб, тамуж, паняцці, зразумеў, меў, аўтар, яны, праўда…). Words identical to Russian must never be in BE_WORDS — «мне», «мой», «была» produced 60 false insertions on 800 live comments before the list was tightened. Shared uk/be words (як, але) prove «not Russian» but not which language → `amb`.
  - **Proper names do not trigger 🐷 on their own.** A capitalized word with ї/є/ґ (Євген, Григорій, Борщагівської — 245 occurrences in a 290-post sample of kievreal1) is a name spelled in Ukrainian orthography, not a мова insertion inside Russian text. If the name is the ONLY evidence, write «имя в украинской орфографии» and do not put 🐷; 🐷 requires a second evidence (another name or a dictionary word). Dictionary words with marker letters (пожеж, станці, районі) remain evidence.
  - **Never add a stem that coincides with a Russian word** (`район`, `станц`, `курсу`, `уходу`, `електротранспорт`): prefix matching catches the Russian word and produces a false hit. Only forms differing from both Russian and Belarusian may be added. Measured on the 290-post sample: false 🐷 on pure Russian text = 0 (every decisive word is a form Russian orthography lacks); the errors went the other way — Ukrainian read as Russian — and the expanded dictionary fixed it (53 `ru` → 254 `uk` / 35 `ru` / 1 `amb`, all 17 reclassified posts checked by hand). The remaining 35 are name-only alert posts this layer cannot decide — that is the method's limit, not a bug. Full rules and numbers: `references/ukrainian-mova-pass.md`.
  - Majority share computed separately for the transcript and for the comments.
- **If the majority is non-Russian** (transcript AND comments) → process as usual, 🐷 for language is NOT applied; the delivery is in Russian (my own translation of the content — no local MT model, say so if machine translation was requested).
- **🐷 is only for Ukrainian.** A Belarusian insertion is NOT tagged 🐷: it is processed as Russian — my translation into Russian, the ordinary passes, toxicity reported without the pig. The script encodes this: `tag = "🐷"` only when a segment's language is `uk`; for `be` and for `amb` the tag is `None`. If the language cannot be decided (only the shared і / shared words як, але), do not guess and do not tag — say the language is uncertain.
- **If the majority is Russian and a fragment is Ukrainian** → insertion: 🐷 + toxicity, two paths, both reported:
  1. **On the original** — the multilingual XLM-R classifier (`scripts/classifier.py`, native uk/be, no translation loss): primary verdict for manipulation devices.
  2. **On my Russian translation** — the Russian-calibrated models (rubert-tiny-toxicity, Kostya165 sarcasm, emotion), because the 🤢/🤮 bands and thresholds are calibrated for Russian.
  If the two paths disagree, show both verdicts — never silently pick one.
- If the insertion is ALSO praise of Ukraine/VСУ/strikes on Russia, 🐷 covers both meanings; the description names which (language insertion, praise, or both). A Belarusian insertion is reported as an insertion with its toxicity verdict, WITHOUT 🐷 (🐷 is only for Ukrainian — same rule as above, the script sets `tag=None` for `be`).
- **Боковой канал (`scripts/side.py`) — мова не смешивается с основным языком без пересборки RAG.** `norm_words` остаётся `[a-zа-яё]`, индекс не перестраивается: боковые слова (і, ї, є, ґ, ў) существуют отдельно и приводятся к русской форме только на стороне запроса (`q_tokens`), чтобы запрос на мове попал в существующий русский индекс. Роль мовы — сигнал (вставка в русскую речь = признак токсичности и враждебности), а не язык корпуса: боковой канал не участвует в scoring корпуса и не меняет то, как индексируются документы. Проверено: индекс не перестроен (sha256 и mtime совпадают), «маніпуляцыя» → «манипуляция» (df 164) вместо фрагмента «пуляцыя» (df 0). Ограничение: приведение — графика плюс окончания, морфологию и перевод оно не делает («найкращий» ≠ «лучший», «маніпуляції» ≠ «манипуляция»); словарного перевода в боковом канале нет.
- Повтор тезиса: основной набор маркеров не трогается, есть отдельный боковой набор из приведённых форм; боковой hit сообщается только когда основной его не покрыл и хотя бы в одном абзаце есть боковое слово (нет двойного счёта). Проверено: основной overlap 2 (miss), боковой 3 → hit через боковой канал. Ограничение: боковой проход ловит только графические варианты одной лексемы — переводные пары («найкращий» ↔ «лучший») он не соединяет.
- In the comments block, insertions are reported as a separate line: count + share of the fetched set, no nicknames (existing rule), with the toxicity verdict per insertion and the language named.
- **Pitfall (recall):** the word regex must include і, ї, є, ґ — they are outside the `а-я` range, so `[а-я]+` splits «дід», «вірус», «Україні» into fragments and word evidence silently stops working. Use `[а-яіїєґў]+`.
- **Verified battery** (run before trusting the gate): Belarusian samples → `be`; Ukrainian controls → `uk`; Russian controls clean, including «наш» and the name «Іван» (single shared і does not fire); short ambiguous case → `amb`, not guessed.

**Neural pass must load mmap-free** (`scripts/neural_pass.py`): the default transformers path mmaps safetensors weights, and under memory pressure on this 16 GB machine macOS cannot fault those pages in — the first Linear layer dies with SIGBUS inside Apple Accelerate BLAS (`cblas_dgemm_singlecore`). Reading the file bytes into RAM (`safetensors.torch.load(open(path,'rb').read())` + `from_config` + `load_state_dict`) avoids the mmap and the forward works even with ~200 MB free. The ONNX classifier path is unaffected (onnxruntime copies the model into its own buffers). `resource_tracker ... leaked semaphore` warnings are shutdown noise, not an accumulating leak. Drop `bert.embeddings.position_ids` when loading state_dict (unexpected key).

Run on every analysis (after fetching comments) — the PRIMARY path is the neural pass (`scripts/neural_pass.py`, psych-rag-env interpreter); the lexicon script is a cross-check only, never the delivered numbers when the neural models are available:

```bash
# cross-check (lexicon):
PYTHONPATH=~/.hermes/cache/scratch/yt-pylib python SKILL_DIR/scripts/comment_sentiment.py "URL" [--json]
```

The script classifies each comment by a Russian emotion lexicon (злость / радость / недовольство / нейтрально) and computes per-thread homogeneity = share of stance-bearing comments in a thread matching the thread's dominant category (proxy for agreement inside the thread; threads with <2 stance-bearing comments are reported as «не измеряется»). Deliver in chat in this shape:

- `Настроение комментариев (N): злость X%, радость Y%, недовольство Z%, нейтрально W%.` — percentages as the script reports them, never recomputed by hand.
- `Однородность в треде: в среднем по тредам N%.` then one line per measurable thread: `@author «первые слова…» — однородность в треде X% (size, доминирует категория)`. If the script also prints `взвешенная по размеру тредов: M%`, that is a DIFFERENT number (size-weighted mean) — quote it only as such, never as «среднее по тредам».
- Thread definition (fixed 2026-10-18): a thread = the root comment + ALL its descendants; a reply-to-a-reply stays in the same thread. When comment ids are not available (flat format from `fetch_comments.py`, which stores only `parent`), the root is derived from the parent prefix before the first dot (`<root_id>.<suffix>` — YouTube comment-id scheme).
- State once, briefly, that the classifier is lexicon-based: sarcasm/irony can be misclassified, and homogeneity measures emotional coloring, not literal semantic agreement.
- Если нейро-слой был **запрошен** и не смог загрузиться (transformers, модели, сеть), доставляемый блок обязан сказать это отдельной строкой: числа получены лексиконом, а не моделью. `comment_analysis.analyze()` несёт флаг `neural_unavailable`, `block()` печатает строку. Когда нейро не запрашивали (явный лексикон для сверки цифр), оговорки нет — тест проверяет оба случая, иначе блок врёт о слое.
- If comments are disabled or fetch fails, say so plainly — never fabricate percentages.
- The replay log record must include a `comments` field: `{mood, homogeneity_overall, threads}` matching the delivered numbers.

## Output Formats

After fetching the transcript, format it based on what the user asks for:

- **Chapters**: Group by topic shifts, output timestamped chapter list
- **Summary**: Concise 5-10 sentence overview of the entire video
- **Chapter summaries**: Chapters with a short paragraph summary for each
- **Thread**: Twitter/X thread format — numbered posts, each under 280 chars
- **Blog post**: Full article with title, sections, and key takeaways
- **Quotes**: Notable quotes with timestamps

## Workflow

0. **Offer, don't assume**: when the user shares a video link without explicit instructions, FIRST ask what they want — e.g. "Сделать краткий пересказ? (рекомендую) / полный транскрипт / главы с таймкодами / разметку тем" — and let them pick. TRIGGER WORD: if the user writes "анализ" (or "проанализируй") together with a YouTube link, this IS the explicit instruction — launch the full analysis WITHOUT asking. OUTPUT ORDER for such a request: (1) the YouTube link on its own first line, (2) then the summary laid out by timecodes (marked timeline with emoji tags and [start–end] ranges, sorted from the start of the video), (3) then the comment block (mood percentages + per-thread homogeneity, see «Comment mood + thread homogeneity»). Deliver in chat/Telegram, no files. For other explicit requests ("сделай транскрипцию", "перескажи") also skip the question and do just that. QUEUE: several analysis requests in one session form a queue — a new link does NOT interrupt the analysis in progress; finish the current one, then work the queue in the order the links arrived. Only the word «срочно» switches to the new task immediately. If a delivery is interrupted mid-way by a new message, the unfinished analysis is still owed — deliver it later in queue order, never drop it. Each video gets its own message starting with its own link; never merge two videos' analyses into one block.
1. **Fetch** the transcript (see Setup; fallback env above). For Russian videos pass `--language ru` explicitly — the helper's default language chain excludes `ru` and reports "No transcript found" even when an auto-generated `ru` track exists. If nothing is found, list tracks with `YouTubeTranscriptApi().list(VIDEO_ID)` and fetch what exists. Run multi-line helper code as a file in scratch, not `python -c` with nested quotes — the shell strips the inner quotes and the code dies with SyntaxError.
2. **Validate**: output must be non-empty and in the expected language. Auto-subtitles contain recognition errors for names/brands — mention that in the answer.
3. **Chunk if needed**: transcripts over ~50K chars — summarize overlapping chunks (~40K with 2K overlap) before merging.
3a. **Psychology layer — brain-hack-analyze on the transcript (standing binding)**: pipe the validated transcript through `scripts/psych_layer.py` (stdin or `--file transcript.txt`). It runs the sibling brain-hack-analyze skill on the transcript ONLY — lexical markers → RAG retrieval → ONNX classifier of persuasion techniques; never applied to the chat message itself. RAG retrieval is queried in 1500-char chunks over the WHOLE transcript (the embedding model itself sees only ~512 tokens, so a single query on a long transcript finds devices only from its beginning — measured: full-text query finds «Ложная дилемма» at position 3739, a 1500-char slice does not). Classifier rows carry `chunk` (which 256-token piece produced the label) and `para` (paragraph index), and paragraphs longer than 256 tokens are processed in pieces instead of being cut. Each flagged paragraph comes back with a `subtype` field computed from the classifier's classes + the marker families of that paragraph (families are bound by the hit's paragraph index, not by substring — a `repetition` hit's `found` is a joined word list that appears in no paragraph): propaganda-family labels (Flag_Waving, Appeal_to_Fear-Prejudice, Guilt_by_Association, Appeal_to_Hypocrisy, Slogans, Appeal_to_Time) → 📢🧠; НЛП-инструментарий из маркерного слоя (пресуппозиционные триггеры по Левинсону, номинализация и удаление агента, модальность/эпистемическая дистанция) → 🔪🧠; остальные метки классификатора (loaded language, doubt, false dilemma, exaggeration, appeal to authority и т.п.) → 🎭🧠. Доверять полю `subtype`; абзацы, отмеченные только НЛП-маркерами без вердикта классификатора, размечаются 🔪🧠 по тем же семьям. Если слой disagrees with the own 3b read, show both verdicts instead of silently picking one. If it returns `unavailable` (venv/index missing), continue with 3b alone — degradation is silent.
3b. **Manipulation pass**: after the keyword pass, re-read the transcript for reasoning devices (substitution of concepts, invalid implications, false dilemmas, hasty generalization, appeal to emotion/authority, survivorship bias, post hoc). Mark them ⚫ followed by a space and the subtype pair: `⚫ 🔪🧠` — НЛП-приёмы (встроенные пресуппозиции, якорение, раппорт, номинализация), `⚫ 🎭🧠` — психологические манипуляции (подмена понятий, ложные дилеммы, апелляции, преувеличения), `⚫ 📢🧠` — пропаганда (кампания/нарратив, вещание на массы, «мы vs они»). With a one-line explanation of the specific logical error, not just the topic.
3c. **Teaching/insider pass**: mark ⭐ segments that teach how something works (mechanisms, engineering, IT, economics, processes) or that contain insider knowledge from a named source/employee. Distinguish from ordinary news restatement and from opinion.
3d. **Praise pass**: scan for explicit positive evaluation of Ukraine, ВСУ or strikes on Russia (including praise quoted from a speaker, e.g. a politician or a commentator). Mark 🐷 and quote the evaluative wording in the description.
3e. **Western idealization pass**: scan for unsupported idealization of the USA/West (ethos as standard, double standards, omitted Western negatives, appeals to «civilization»). Mark 🦅 and name the specific unsupported claim or double standard.
3f. **Pro-Russian hype pass**: scan for unfounded praise of Russia/ВС РФ («гойда», гордость, «мы великие», бездоказательный героизм, «наши vs не наши», замалчивание плохого в России). Mark 🐻 and name the specific device and what evidence is missing.
3g. **Infocyan pass**: in economic content, check each claim against basic economic theory (risk/return, guaranteed vs non-guaranteed income, dividends vs free cash flow, liquidity, survivorship bias, correlation vs causation). Mark 🦹 for claims that contradict theory and for sales of the author's own economics/stock-market courses, newsletters, signals or consultations.
3h. **Marketing pass** (модификатор 💸): перечитай транскрипт на продающие приёмы — CTA, искусственный дефицит/срочность (FOMO), ценовое якорение, социальное доказательство с непроверяемыми цифрами, лженаучные апелляции без источника, подарки/бонусы за немедленную покупку. 💸 добавляется в КОНЕЦ строки как модификатор к основному тегу фрагмента (🟤💸, ⭐💸, 🔴💸); если продающий язык — единственное содержание фрагмента, строка несёт один 💸. В описании — конкретный приём и цитата.
4. **Transform** into the requested format; default is a summary + timestamped chapters, in Russian (user reads Russian).
5. **Mark** fragments with the tag system below (user's standing preference — apply without being asked).
6. **Deliver in chat, not on disk**: the user sends videos to quickly learn their contents and will NOT open files. ALWAYS print the summary, chapters, and the marked timeline directly in the chat response. Never write transcript/summary files unless the user explicitly asks. Rich formatting (colors) needs an HTML file — only create one if the user asks for it.
7. **Replay log (the one exception)**: after every analysis, append one compact record to `~/.hermes/video_summaries.jsonl` (persistent, NOT scratch — scratch is pruned after 24h): `{"url", "title", "date", "summary", "timeline"}` (plus `comments` for the mood/homogeneity block) — take the title from the oEmbed endpoint (`https://www.youtube.com/oembed?url=<URL>&format=json`), since the transcript helper returns no title — timeline as the final sorted marked list with emoji tags and [start–end] ranges. When the user (usually from phone/Telegram) asks about a video from history — «какой был последний запрос», «повтори последнее саммари», «что за видео» — the reply MUST contain all three parts for each entry (default last 2, newest last): (1) YouTube link, (2) the summary, (3) the full tagged timeline. Never send only the link or only the topic. DELIVERY TARGET: the user asks these questions from their phone, so the answer must reach the Telegram bot chat — send ONLY the info about the requested video (link + summary + tagged timeline), no extra commentary, no file paths, no re-fetching. This applies even when the video is sent from the terminal/CLI: still post the same formatted message to the Telegram bot chat (the user may want to download and listen to the video). If the gateway is connected, the normal chat reply already lands in Telegram; when acting from another session, post it via the Bot API (token and allowlisted chat_id read from the Hermes env file). Split BEFORE sending, not after a failure: `sendMessage` returns HTTP 400 Bad Request when the text exceeds 4096 chars, so build the parts first (summary cut at sentence boundaries, timeline accumulated line by up to ~4090) and post them sequentially. Measure the assembled message programmatically before sending: if it is only slightly over the limit, shorten the descriptions of the longest timeline entries (keep emoji, range and the named device) rather than splitting — one message is what the user reads on the phone. The log record must match the version that was actually delivered. When splitting, post the FIRST part via the Bot API and deliver the LAST part as the chat reply: the Bot API posts immediately while the chat reply lands at the end of the turn, so posting part 2 first arrives out of order. Never duplicate content the user already received — the chat reply is either the remaining part or a short confirmation naming where it starts. Read the token from `~/.hermes/.env`; when the bot runs through a webhook the API returns `ok: true` with no `message_id`, so report the delivery status honestly instead of inventing ids.
8. **Verify**: re-read the output for coherence, correct timestamps, completeness, and that the timeline is sorted.

## Marking system (user-defined, standing preference)

Apply to every video analysis. Format: `EMOJI [MM:SS–MM:SS] краткое описание`, timestamp range start–end right after the emoji. In Telegram delivery the range is a clickable Markdown link to the video at the period start: `EMOJI [MM:SS–MM:SS](https://www.youtube.com/watch?v=<VIDEO_ID>&t=<start_seconds>)`, start_seconds computed from the range start. The 4096 limit counts the full Markdown text including URLs — re-split accordingly. The log record keeps plain ranges (the link is mechanically derivable from url + start time). If a message was already sent without links, edit it via editMessageText (Bot API, chat_id + message_id from the send response).

| Tag | Category |
|-----|----------|
| 🔴 | Явный оценочный негатив БЕЗ фактической базы. Критерии: (1) в фрагменте есть оценка плохого состояния/качества, выраженная оценочной лексикой (преувеличение, эмоциональная характеристика), а не конкретными цифрами или наблюдаемым фактом; (2) нет уверенного манипулятивного/НЛП-вердикта (иначе строка — только пара подтипа); (3) нет более специфичного тега. Фрагменты с реальными данными (цена, пробег, толщина слоя, механизм поломки) НЕ маркируются 🔴 — если они ещё и объясняют механизм, это ⭐, иначе тег не ставится вообще и фрагмент называется в строке «фактические фрагменты без тегов». Таймлайн не должен превращаться в непрерывную череду 🔴/🎭🧠: тег ставится только там, где приём или оценочная лексика реально есть. Критика России (армия, правительство, институты власти) остаётся 🔴 |
| 🍆 | Явная анти-российская пропаганда — сильнейшая категория, выше простой критики; обычная критика остаётся 🔴 |
| 🍏 | Позитив о России: прогресс России, успехи армии, победы на поле боя, другие позитивные мысли о России |
| 🟢 | Прогресс человечества: наука, производство, роботы, ИИ, мировая инфраструктура |
| 🟤 | Реклама: отдельные блоки И нативная (спонсорские упоминания, промо продуктов/медиа, продвижение собственных продуктов автора — newsletter, книга, платный коучинг). Простой самопиар канала («подписывайтесь», like/subscribe) — НЕ реклама и не маркируется. Если рекламы нет — сказать об этом явно |
| 🟣 | Фрагменты вне общего контекста видео (отступления, внутренние шутки, метa-разговор), НЕ являющиеся рекламой и НЕ содержащие анти-российского подтекста |
| 🎭🧠 / 🔪🧠 / 📢🧠 | Психологические манипуляции (⚫ удалён по команде пользователя — чёрный смайлик больше не используется). Пара подтипа ставится САМА, без ⚫ и без других тегов в этой строке: 🔪🧠 — НЛП-приёмы (пресуппозиции, якорение, номинализация), 🎭🧠 — манипуляции (подмена понятий, софистика, ложные импликации, ложные дилеммы, апелляции, преувеличения), 📢🧠 — пропаганда. Явно называть приём и объяснять, в чём именно ошибка. Ставится только при уверенном вердикте: p ≥ порога метки из thresholds.npy И p ≥ 0.70, И в фрагменте реально назван конкретный логический приём. Если уверенного вердикта нет — это не манипуляция. |
| 💸 | МОДИФИКАТОР «написано маркетинговым языком»: добавляется в КОНЕЦ строки к основному тегу (🟤💸 — реклама продающим языком, ⭐💸 — обучение в стиле распродажи, 🔴💸 — негатив продающим языком) либо стоит один, если продающий язык — единственное содержание фрагмента. Маркетинговые приёмы: призыв к действию (CTA), искусственный дефицит/срочность (FOMO), ценовое якорение («обычная цена X — для вас Y»), социальное доказательство с непроверяемыми цифрами («выбор 90% покупателей»), лженаучная апелляция («учёные доказали» без источника), приманка-подарок/бонус за немедленную покупку. В описании называть конкретный приём и цитату |
| ⭐ | Обучающая техническая информация и технические инсайды: объяснение, как что-то устроено/работает (механизмы, инженерия, IT, экономика, процессы), и инсайдерские детали — то, что рассказал источник изнутри, чего нет в публичных данных. В описании фрагмента указывать, чему именно учит фрагмент или какой это инсайд |
| 🐷 | Явное расхваливание Украины, ВСУ и ударов по России: похвала ударов/атак, оправдание обстрелов, восхваление ВСУ, «молодцы», «правильно делают», «так и надо», позитивная оценка разрушений в России. В описании указывать, что именно хвалится |
| 🦅 | Необоснованное манипулятивное восхваление США и западного мира: подача Запада как эталона («там безопаснее/чище/справедливее») без аргументов и без учёта контраргументов, двойные стандарты (плохое у Запада замалчивается, плохое в России подчёркивается), идеализация институтов/общества США и ЕС, апелляция к «цивилизованности». В описании указывать, в чём именно необоснованность или двойной стандарт |
| 🐻 | Необоснованное про-российское восхваление: «гойда», гордость «мы великие», хвалебные оценки ВС РФ и России без конкретных фактов, приёмы про-российской психологической манипуляции (апелляция к гордости/страху, подвиг без доказательств, «наши» vs «не наши», замалчивание плохого в России при подчёркнутом плохом на Западе/Украине). В описании указывать, какой именно приём используется и чем подтверждено (или не подтверждено) |
| 🦹 | Инфоцыганщина в экономическом контенте: тезисы, противоречащие экономической теории (доходность «гарантирована», «рынок всегда растёт», «дивиденды = гарантированная прибыль», «высокая доходность без риска», подмена понятий в финансах), плюс реклама собственных курсов/рассылок/консультаций по экономике и фондовому рынку. В описании указывать, какой тезис противоречит теории и что именно продаётся |
| 🔵 | Скрываемая тайна / «правда», которую раскрывают: «на самом деле», «настоящая причина», инсайдерские знания, замалчиваемые факты |
| ☠️ | Смерть или суицид в контексте: прямое упоминание смерти человека, суицида, «на гробовые», угроза жизни как вывод автора. Применяется и в таймлайне, и в блоке комментариев. Косы в Unicode нет — закреплён ☠️ (выбор пользователя, менять самостоятельно нельзя). Смерть техники («умер мотор», «машина умерла») — НЕ ☠️ |
| 🤢 / 🤮 | Токсичность и сарказм по моделям `cointegrated/rubert-tiny-toxicity` и `Kostya165/rubert_tiny2_russian_emotion_sentiment`. 🤢 — малая степень: сарказм/насмешка без прямого оскорбления, или токсичность в диапазоне p 0.30–0.70. 🤮 — высокая токсичность: p ≥ 0.70 по insult/obscenity/threat, или несколько токсичных меток сразу. Применяется в таймлайне и в блоке комментариев. Глифа «рыгающий зелёной кислотой» в Unicode нет — это выбор пользователя, зафиксированный как 🤢/🤮 |

Rules:
- **Reconcile with what was already delivered**: when the user challenges a contradiction between two answers about the same video, check the actual artifact (scratch transcript, log record) and say which path was used for THAT video before restating the general rule. A categorical claim that contradicts the analysis you delivered (e.g. "Whisper never touches video" after a Whisper-based analysis) reads as an error — phrase the pipeline as an order: existing subtitle track first, Whisper only as the audio-track fallback when no captions exist.
- **Compaction pitfall**: context compression prunes this skill's content from context — the tag table is gone, and a timeline delivered after compaction comes out untagged (plain `[MM:SS]` lines). If the session shows a compaction summary or a `[SKILL_PRUNED]` marker, reload this skill with `skill_view(name='youtube-content')` BEFORE producing any analysis delivery, even if the transcript is already in scratch and no re-fetch is needed. Never deliver a timeline that is not the marked timeline.
- **Status post (standing user rule, 2026-10-18, v2)**: at analysis start, IMMEDIATELY send ONE status post and never spam progress messages. Use `scripts/status_post.py` through `terminal`: `python3 <script> start <chat_id=362122320> <key> <preview> <status>` — preview = video link + title ( ALWAYS kept at the TOP of the post, the script stores and re-renders it on every edit); key = video ID. On EVERY meaningful step (transcript fetched, comments fetched, neural pass running, assembling output) run `update <key> <new status>` — it edits the SAME post (`editMessageText`) keeping the preview on top. After delivering the final output, run `done <key>` (sets «Готово»). The final analysis itself is delivered exactly as before — the status post only tracks progress. The helper must survive API errors: wrap the Bot API call in try/except, retry on 429 (editMessageText limit ~3 edits/s, ~20/min per chat) and on network failure, and guard the 4096-char limit for editMessageText — an unhandled HTTPError kills the post mid-analysis, and the status post is the only allowed progress channel.
- **NO interim chat text during an analysis turn (правило пользователя, 2026-10-18)**: every assistant text message lands in Telegram as a chat message — writing progress commentary between tool calls IS the spam the user complained about. During the whole analysis turn, emit ONLY tool calls (all progress goes into status-post edits); the first chat text of the turn must be the delivered final output. If the analysis is short and already done before the first update, still send start + done around the delivery.
- **Long runs**: if a step takes >10 minutes, keep updating the status post (heartbeat every ~5 min, e.g. «нейро-классификация: 800/1348») instead of writing new chat messages.
- **Один основной тег на строку + модификатор 💸**: строка несёт ровно один основной тег (пара подтипа считается одним тегом) и, при продающем языке, модификатор 💸 в конце (🟤💸, ⭐💸, 🔴💸). Комбинации вида «🔵+🔴» запрещены. Если продающий язык — единственное содержание фрагмента, строка несёт один 💸 без основного тега. Приоритет выбора ОСНОВНОГО тега: уверенный манипулятивный/НЛП-вердикт → специфичный контент-тег (☠️, 🐷, 🦅, 🐻, 🦹, 🍆, 🍏, 🟢, ⭐, 🔵, 🟣, 🟤) → 🔴 как резерв.
- **SORTING**: output ALL marked fragments (any tag — критика, реклама и т.д.) as ONE timeline sorted by start time from the beginning of the video; never group by tag/category first. Long ranges — cut precisely so the timeline stays monotonic.
- **Manual review**: build marks with a keyword pass first, then review every marked sentence manually; keywords like «пострадало», «закон», «донос» match technical details, other countries, and non-Russian contexts — unmark those.
- Neutral politics (parties, laws, elections as facts) is NOT marked.
- ⚫ requires an explicit call-out in the fragment description: name the device and the specific error, e.g. «подмена: X выдаётся за Y», «ложная импликация: из A не следует B», «ложная дилемма: только A или B», «апелляция к страху». Format: `⚫ 🔪🧠 [MM:SS–MM:SS] …` for manipulation/NLP devices and `⚫ 📢🧠 [MM:SS–MM:SS] …` for propaganda; ⚫ alone is an incomplete mark. If a fragment is both content-critical and manipulative, put ⚫ in the timeline and mention the other category in the description (e.g. `⚫ 🔪🧠 [MM:SS–MM:SS] 🔴-критика + подмена: …`).
- ⭐ marks teaching/insider value, not opinion: the fragment must actually explain a mechanism or convey insider knowledge. In the description state what is taught («⭐ объясняет, как кассовый разрыв превращается в цепную реакцию») or what the insider detail is («⭐ инсайд: за 7 лет потеряно древо кода, идёт откат»). If a ⭐ fragment also falls into another category, keep ⭐ in the timeline slot and note the other tag in the description (e.g. `⭐ [MM:SS–MM:SS] 🟢-прогресс: …`).
- 🐷 is for explicit praise, not neutral reporting: a factual mention of attacks or a neutral description of damage is NOT 🐷. Mark only where the author or a quoted speaker positively evaluates Ukraine/VСУ or the strikes on Russia. If the praise is also explicit anti-Russian propaganda, use 🐷 (more specific) and note 🍆 in the description. If the praise is delivered through a logical device, note ⚫ in the description.
- 🦅 is for idealization without argument, not for any positive mention of the West: factual data about Western institutions, neutral comparison with concrete numbers, or criticism of the West is NOT 🦅. Mark when the praise is unsupported or double-standard (e.g. «там всё удобно и безопасно» без фактов; плохое в западных странах замалчивается при подчёркнутом плохом в России). If the idealization works through a logical device, note ⚫ in the description.
- 🍏 vs 🐻: 🍏 is positive about Russia backed by concrete facts (progress, results, numbers, verified successes); 🐻 is praise without facts — emotional appeal, «гойда», гордость, бездоказательный героизм, «мы/они». If a fragment has both (fact + emotional framing), mark 🐻 and note 🍏 in the description.
- 🦹 is for economically wrong claims sold as truth, not for any economic statement: correct analysis with numbers and caveats is NOT 🦹. Mark when a claim contradicts basic economic theory (risk-free high yield, guaranteed dividends, guaranteed growth, survivorship-bias «я заработал, значит метод работает») or when the author sells his own course/newsletter/consulting on economics or the stock market. If the fragment is also plain advertising, use 🦹 and note 🟤 in the description.
- Distinguish genuine argument from manipulation: a claim that is simply false or one-sided is not automatically ⚫ — ⚫ is for the reasoning device itself (invalid inference, substitution of concepts, emotional pressure). Do not mark every disagreeable statement as ⚫.
- The delivered timeline IS the marked timeline: every entry carries its emoji tag. A plain timestamped list with no tags is an incomplete analysis — run the passes over the whole transcript before writing, not only over the political segments.
- Close every analysis with a line naming the tags that did not occur (e.g. «Нет 🐷, 🦹, 🔵»), same as `brain-hack-analyze` — it separates "not present" from "not noticed". For content with no political/economic claims at all (vlogs, tutorials, repair logs), say that plainly instead of forcing tags.
- Telegram renders no arbitrary text colors — the emoji system above IS the highlighting there. An HTML file only if the user explicitly asks for colors.
- If the user asks for an emoji that does not exist in Unicode (a recolored or partial glyph, e.g. «жёлтый орёл», «только голова орла»), say the constraint plainly and offer the closest real glyphs instead of silently substituting one; keep the user's chosen tag exactly as agreed and do not change it later on your own.
- Deletion sync: when the user deletes a video message in Telegram, remove that video's record from `~/.hermes/video_summaries.jsonl` on the next turn. Caveat to state honestly: a Telegram bot cannot see deletions in private chats (the Bot API delivers no message_deleted event for DMs), so this works only when the user tells me they deleted it (or the deletion shows up as a channel_post_deleted in a channel). Per-message rows in the session history DB cannot be surgically deleted — only the summary log can.

## Error Handling

- **Transcript disabled**: tell the user; suggest they check if subtitles are available on the video page.
- **Private/unavailable video**: relay the error and ask the user to verify the URL.
- **No matching language**: list what actually exists with `YouTubeTranscriptApi().list(VIDEO_ID)` (fresh API instance, v1.x) before concluding anything; auto-generated tracks (`is_generated: True`) are fetchable like manual ones. Note the actual language found to the user.
- **Dependency missing**: repeat PM preparation and reactivation above, then verify the helper uses that Python. Do not repair the selected generation with pip.
