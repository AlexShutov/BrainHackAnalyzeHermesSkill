---
name: youtube-content
description: "YouTube transcripts to summaries, threads, blogs."
version: 2.3.0
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

## Comments (no API key needed)

Comments are fetchable WITHOUT a YouTube Data API key via the innertube endpoint through yt-dlp. The official Data API (`commentThreads`) requires an API key — not configured here; do not guess one. Install yt-dlp into the same isolated target dir as the transcript lib:

```bash
python -m pip -q install --target ~/.hermes/cache/scratch/yt-pylib yt-dlp
PYTHONPATH=~/.hermes/cache/scratch/yt-pylib python SKILL_DIR/scripts/fetch_comments.py "URL" [--limit N] [--json]
```

Fields per comment: `author, text, likes, timestamp (unix), pinned, parent` (`parent == "root"` = top-level; otherwise the parent comment id — replies). Order is YouTube's default (top). Pitfalls:
- Comments disabled on the video → script exits with a clear message; say so, don't fabricate.
- Fetching ALL comments on a big video is slow — use `--limit` for a quick pass (e.g. top 50).
- The WEB innertube client now returns key-based commentViewModels without text (raw scraping returns 0 comments); yt-dlp handles this correctly — do not hand-roll innertube scraping.
- Comments are part of the standard analysis: on any «анализ»/«проанализируй» request, fetch ALL comments and run `scripts/comment_sentiment.py` — deliver the mood + homogeneity block after the timeline (see below). For plain summary/transcript requests without the trigger word, comments are not fetched.

## Comment mood + thread homogeneity (standing algorithm)

Run on every analysis (after fetching comments):

```bash
PYTHONPATH=~/.hermes/cache/scratch/yt-pylib python SKILL_DIR/scripts/comment_sentiment.py "URL" [--json]
```

The script classifies each comment by a Russian emotion lexicon (злость / радость / недовольство / нейтрально) and computes per-thread homogeneity = share of stance-bearing comments in a thread matching the thread's dominant category (proxy for agreement inside the thread; threads with <2 stance-bearing comments are reported as «не измеряется»). Deliver in chat in this shape:

- `Настроение комментариев (N): злость X%, радость Y%, недовольство Z%, нейтрально W%.` — percentages as the script reports them, never recomputed by hand.
- `Однородность в треде: в среднем по тредам N%.` then one line per measurable thread: `@author «первые слова…» — однородность в треде X% (size, доминирует категория)`.
- State once, briefly, that the classifier is lexicon-based: sarcasm/irony can be misclassified, and homogeneity measures emotional coloring, not literal semantic agreement.
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
3a. **Psychology layer — bigbro-analyze on the transcript (standing binding)**: pipe the validated transcript through `scripts/psych_layer.py` (stdin or `--file transcript.txt`). It runs the sibling bigbro-analyze skill on the transcript ONLY — lexical markers → RAG retrieval → ONNX classifier of persuasion techniques; never applied to the chat message itself. Merge its verdicts into pass 3b: a paragraph the layer flags gets ⚫ with the subtype pair — propaganda-family labels (Name Calling, Guilt by Association, Flag Waving, Appeal to Fear/Prejudice inside a mass narrative, «мы vs они») → `⚫ 📢🧠`; personal manipulation / NLP devices (loaded language, doubt, false dilemma, presupposition triggers, anchoring, rapport tricks) → `⚫ 🔪🧠`. If the layer disagrees with the own 3b read, show both verdicts instead of silently picking one. If it returns `unavailable` (venv/index missing), continue with 3b alone — degradation is silent.
3b. **Manipulation pass**: after the keyword pass, re-read the transcript for reasoning devices (substitution of concepts, invalid implications, false dilemmas, hasty generalization, appeal to emotion/authority, survivorship bias, post hoc). Mark them ⚫ followed by a space and the subtype pair: `⚫ 🔪🧠` for psychological manipulation / NLP devices (incl. встроенные пресуппозиции, якорение, раппорт), `⚫ 📢🧠` for propaganda-style manipulation (кампания/нарратив, вещание на массы, «мы vs они»). With a one-line explanation of the specific logical error, not just the topic.
3c. **Teaching/insider pass**: mark ⭐ segments that teach how something works (mechanisms, engineering, IT, economics, processes) or that contain insider knowledge from a named source/employee. Distinguish from ordinary news restatement and from opinion.
3d. **Praise pass**: scan for explicit positive evaluation of Ukraine, ВСУ or strikes on Russia (including praise quoted from a speaker, e.g. a politician or a commentator). Mark 🐷 and quote the evaluative wording in the description.
3e. **Western idealization pass**: scan for unsupported idealization of the USA/West (ethos as standard, double standards, omitted Western negatives, appeals to «civilization»). Mark 🦅 and name the specific unsupported claim or double standard.
3f. **Pro-Russian hype pass**: scan for unfounded praise of Russia/ВС РФ («гойда», гордость, «мы великие», бездоказательный героизм, «наши vs не наши», замалчивание плохого в России). Mark 🐻 and name the specific device and what evidence is missing.
3g. **Infocyan pass**: in economic content, check each claim against basic economic theory (risk/return, guaranteed vs non-guaranteed income, dividends vs free cash flow, liquidity, survivorship bias, correlation vs causation). Mark 🦹 for claims that contradict theory and for sales of the author's own economics/stock-market courses, newsletters, signals or consultations.
4. **Transform** into the requested format; default is a summary + timestamped chapters, in Russian (user reads Russian).
5. **Mark** fragments with the tag system below (user's standing preference — apply without being asked).
6. **Deliver in chat, not on disk**: the user sends videos to quickly learn their contents and will NOT open files. ALWAYS print the summary, chapters, and the marked timeline directly in the chat response. Never write transcript/summary files unless the user explicitly asks. Rich formatting (colors) needs an HTML file — only create one if the user asks for it.
7. **Replay log (the one exception)**: after every analysis, append one compact record to `~/.hermes/video_summaries.jsonl` (persistent, NOT scratch — scratch is pruned after 24h): `{"url", "title", "date", "summary", "timeline"}` (plus `comments` for the mood/homogeneity block) — take the title from the oEmbed endpoint (`https://www.youtube.com/oembed?url=<URL>&format=json`), since the transcript helper returns no title — timeline as the final sorted marked list with emoji tags and [start–end] ranges. When the user (usually from phone/Telegram) asks about a video from history — «какой был последний запрос», «повтори последнее саммари», «что за видео» — the reply MUST contain all three parts for each entry (default last 2, newest last): (1) YouTube link, (2) the summary, (3) the full tagged timeline. Never send only the link or only the topic. DELIVERY TARGET: the user asks these questions from their phone, so the answer must reach the Telegram bot chat — send ONLY the info about the requested video (link + summary + tagged timeline), no extra commentary, no file paths, no re-fetching. This applies even when the video is sent from the terminal/CLI: still post the same formatted message to the Telegram bot chat (the user may want to download and listen to the video). If the gateway is connected, the normal chat reply already lands in Telegram; when acting from another session, post it via the Bot API (token and allowlisted chat_id read from the Hermes env file). Split BEFORE sending, not after a failure: `sendMessage` returns HTTP 400 Bad Request when the text exceeds 4096 chars, so build the parts first (summary cut at sentence boundaries, timeline accumulated line by up to ~4090) and post them sequentially. Measure the assembled message programmatically before sending: if it is only slightly over the limit, shorten the descriptions of the longest timeline entries (keep emoji, range and the named device) rather than splitting — one message is what the user reads on the phone. The log record must match the version that was actually delivered. When splitting, post the FIRST part via the Bot API and deliver the LAST part as the chat reply: the Bot API posts immediately while the chat reply lands at the end of the turn, so posting part 2 first arrives out of order. Never duplicate content the user already received — the chat reply is either the remaining part or a short confirmation naming where it starts. Read the token from `~/.hermes/.env`; when the bot runs through a webhook the API returns `ok: true` with no `message_id`, so report the delivery status honestly instead of inventing ids.
8. **Verify**: re-read the output for coherence, correct timestamps, completeness, and that the timeline is sorted.

## Marking system (user-defined, standing preference)

Apply to every video analysis. Format: `EMOJI [MM:SS–MM:SS] краткое описание`, timestamp range start–end right after the emoji. In Telegram delivery the range is a clickable Markdown link to the video at the period start: `EMOJI [MM:SS–MM:SS](https://www.youtube.com/watch?v=<VIDEO_ID>&t=<start_seconds>)`, start_seconds computed from the range start. The 4096 limit counts the full Markdown text including URLs — re-split accordingly. The log record keeps plain ranges (the link is mechanically derivable from url + start time). If a message was already sent without links, edit it via editMessageText (Bot API, chat_id + message_id from the send response).

| Tag | Category |
|-----|----------|
| 🔴 | Критика России: армии, правительства, институтов власти (Госдума, ФСБ, Минцифры и т.п.); скрытый подтекст, критикующий Россию; манипулятивные сопоставления без логической импликации, выставляющие Россию в плохом свете (напр. «Amazon не работает как магазин, а Wildberries работает»). Если в видео нет ничего негативного о России — 🔴 используется для общего негатива/токсичности, и об отсутствии России не упоминается вовсе |
| 🍆 | Явная анти-российская пропаганда — сильнейшая категория, выше простой критики; обычная критика остаётся 🔴 |
| 🍏 | Позитив о России: прогресс России, успехи армии, победы на поле боя, другие позитивные мысли о России |
| 🟢 | Прогресс человечества: наука, производство, роботы, ИИ, мировая инфраструктура |
| 🟤 | Реклама: отдельные блоки И нативная (спонсорские упоминания, промо продуктов/медиа, продвижение собственных продуктов автора — newsletter, книга, платный коучинг). Простой самопиар канала («подписывайтесь», like/subscribe) — НЕ реклама и не маркируется. Если рекламы нет — сказать об этом явно |
| 🟣 | Фрагменты вне общего контекста видео (отступления, внутренние шутки, метa-разговор), НЕ являющиеся рекламой и НЕ содержащие анти-российского подтекста |
| ⚫ | Психологические манипуляции: подмена понятий, софистика, логически неверные импликации, ложные дилеммы, апелляция к эмоциям/авторитету, подмена причинно-следственных связей. В таймлайне после ⚫ и пробела ставится пара подтипа: 🔪🧠 — манипуляции и НЛП-приёмы, 📢🧠 — пропаганда. Явно называть приём и объяснять, в чём именно ошибка |
| ⭐ | Обучающая техническая информация и технические инсайды: объяснение, как что-то устроено/работает (механизмы, инженерия, IT, экономика, процессы), и инсайдерские детали — то, что рассказал источник изнутри, чего нет в публичных данных. В описании фрагмента указывать, чему именно учит фрагмент или какой это инсайд |
| 🐷 | Явное расхваливание Украины, ВСУ и ударов по России: похвала ударов/атак, оправдание обстрелов, восхваление ВСУ, «молодцы», «правильно делают», «так и надо», позитивная оценка разрушений в России. В описании указывать, что именно хвалится |
| 🦅 | Необоснованное манипулятивное восхваление США и западного мира: подача Запада как эталона («там безопаснее/чище/справедливее») без аргументов и без учёта контраргументов, двойные стандарты (плохое у Запада замалчивается, плохое в России подчёркивается), идеализация институтов/общества США и ЕС, апелляция к «цивилизованности». В описании указывать, в чём именно необоснованность или двойной стандарт |
| 🐻 | Необоснованное про-российское восхваление: «гойда», гордость «мы великие», хвалебные оценки ВС РФ и России без конкретных фактов, приёмы про-российской психологической манипуляции (апелляция к гордости/страху, подвиг без доказательств, «наши» vs «не наши», замалчивание плохого в России при подчёркнутом плохом на Западе/Украине). В описании указывать, какой именно приём используется и чем подтверждено (или не подтверждено) |
| 🦹 | Инфоцыганщина в экономическом контенте: тезисы, противоречащие экономической теории (доходность «гарантирована», «рынок всегда растёт», «дивиденды = гарантированная прибыль», «высокая доходность без риска», подмена понятий в финансах), плюс реклама собственных курсов/рассылок/консультаций по экономике и фондовому рынку. В описании указывать, какой тезис противоречит теории и что именно продаётся |
| 🔵 | Скрываемая тайна / «правда», которую раскрывают: «на самом деле», «настоящая причина», инсайдерские знания, замалчиваемые факты |

Rules:
- **Reconcile with what was already delivered**: when the user challenges a contradiction between two answers about the same video, check the actual artifact (scratch transcript, log record) and say which path was used for THAT video before restating the general rule. A categorical claim that contradicts the analysis you delivered (e.g. "Whisper never touches video" after a Whisper-based analysis) reads as an error — phrase the pipeline as an order: existing subtitle track first, Whisper only as the audio-track fallback when no captions exist.
- **Compaction pitfall**: context compression prunes this skill's content from context — the tag table is gone, and a timeline delivered after compaction comes out untagged (plain `[MM:SS]` lines). If the session shows a compaction summary or a `[SKILL_PRUNED]` marker, reload this skill with `skill_view(name='youtube-content')` BEFORE producing any analysis delivery, even if the transcript is already in scratch and no re-fetch is needed. Never deliver a timeline that is not the marked timeline.
- **Long runs**: if an analysis (or any step of it) takes more than 10 minutes, post a one-line
  status to the chat — the user assumes the machine has hung otherwise.
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
- Close every analysis with a line naming the tags that did not occur (e.g. «Нет 🐷, 🦹, 🔵»), same as `bigbro-analyze` — it separates "not present" from "not noticed". For content with no political/economic claims at all (vlogs, tutorials, repair logs), say that plainly instead of forcing tags.
- Telegram renders no arbitrary text colors — the emoji system above IS the highlighting there. An HTML file only if the user explicitly asks for colors.
- If the user asks for an emoji that does not exist in Unicode (a recolored or partial glyph, e.g. «жёлтый орёл», «только голова орла»), say the constraint plainly and offer the closest real glyphs instead of silently substituting one; keep the user's chosen tag exactly as agreed and do not change it later on your own.
- Deletion sync: when the user deletes a video message in Telegram, remove that video's record from `~/.hermes/video_summaries.jsonl` on the next turn. Caveat to state honestly: a Telegram bot cannot see deletions in private chats (the Bot API delivers no message_deleted event for DMs), so this works only when the user tells me they deleted it (or the deletion shows up as a channel_post_deleted in a channel). Per-message rows in the session history DB cannot be surgically deleted — only the summary log can.

## Error Handling

- **Transcript disabled**: tell the user; suggest they check if subtitles are available on the video page.
- **Private/unavailable video**: relay the error and ask the user to verify the URL.
- **No matching language**: list what actually exists with `YouTubeTranscriptApi().list(VIDEO_ID)` (fresh API instance, v1.x) before concluding anything; auto-generated tracks (`is_generated: True`) are fetchable like manual ones. Note the actual language found to the user.
- **Dependency missing**: repeat PM preparation and reactivation above, then verify the helper uses that Python. Do not repair the selected generation with pip.
