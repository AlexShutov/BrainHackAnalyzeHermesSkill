# Rutube plumbing depth

Load when fetching Rutube transcripts, ranking a channel, or running long-audio transcription.

## Transcribing a video with no captions

- `scripts/rutube_transcript.py URL` fetches subtitles only; when the extractor reports "has no
  subtitles", fall back to faster-whisper (`~/.hermes/tools/stt-env/bin/python`, model `small`,
  CPU int8) — flag the result as a machine transcript.
- **No audio-only stream**: `-f bestaudio` fails ("Requested format is not available") — Rutube
  serves combined A/V m3u8 streams only. Take the lowest-bitrate format id from `--list-formats`
  and transcribe the downloaded file directly (PyAV decodes mp4).
- Quote yt-dlp output templates (`"out.%(ext)s"`) — unquoted parentheses break the shell.
- Resource profile (measured): ~1 GB peak RSS, ~10× realtime on CPU → a 4-hour video ≈ 25 min.
  Memory does not grow with audio length (30 s windows), so long videos are worth running, not
  skipping.
- Run long jobs as a script file in the background with `notify` on a completion marker; a
  multi-minute heredoc in a foreground call times out and loses the kernel state. Confirm the
  download produced a file before feeding it to the transcriber — a glob that assumes the download
  landed raises FileNotFoundError.

## Ranking a channel by popularity

`https://rutube.ru/api/video/person/<channel_id>/?limit=50&format=json`:
- View count is the `hits` field — `view_count` is 0/absent on this endpoint.
- No sort parameter works (`sort=hits`, `sort=popular` are silently ignored): fetch every page via
  `next` and rank in code.
- Exclude the live-channel entry: it appears in the same listing with `duration: 0` and millions of
  `hits` and is the stream, not a video.
- Channels carry thousands of videos: paginate in batches, appending to a scratch JSON after each
  call, with a per-page timeout and retry. If the kernel state is lost, rebuild page numbers as
  `?page=N` from the number of rows already saved.
- Per-page result counts can be smaller than `limit`; trust the accumulated rows, not `limit × pages`.
